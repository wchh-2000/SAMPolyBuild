"""Thin Prompt inference adapter for tile-review click additions."""

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable
import cv2
import numpy as np

from tools.point_refine_utils import (
    extract_main_polygon_from_mask,
    keep_primary_component,
    refine_mask_by_prompt_points,
    save_binary_mask,
    save_point_refine_regions,
    save_prompt_points_json,
    split_labeled_points,
)
from utils.negative_feature_refine import refine_mask_by_negative_features, save_negative_feature_debug

_PROMPT_RUNTIME_CACHE = {}


@dataclass
class PromptSessionCache:
    """Cache one tile image and its active crop embedding during Refine."""

    image_path: Path | None = None
    image: Any = None
    embedding_key: tuple | None = None
    _embedding_owner: int | None = None
    prepare_count: int = 0
    debug_step: int = 0

    def get_image(self, image_path: Path, loader: Callable[[Path], Any]):
        """Load an RGB tile once per normalized image path."""
        normalized = Path(image_path).expanduser().resolve()
        if self.image_path != normalized or self.image is None:
            self.image = loader(normalized)
            self.image_path = normalized
            self.embedding_key = None
            self._embedding_owner = None
        return self.image

    def ensure_embedding(
        self,
        predictor,
        key: tuple,
        prepare_embedding: Callable[[], None],
    ) -> bool:
        """Prepare an embedding only when its predictor or crop key changes."""
        owner = id(predictor)
        if self.embedding_key == key and self._embedding_owner == owner:
            return False
        prepare_embedding()
        self.embedding_key = key
        self._embedding_owner = owner
        self.prepare_count += 1
        return True

    def clear(self) -> None:
        """Release cached image references when a session exits."""
        self.image_path = None
        self.image = None
        self.embedding_key = None
        self._embedding_owner = None
        self.debug_step = 0

    def next_debug_step(self) -> int:
        """Return the next 1-based debug step number for one active session."""
        self.debug_step += 1
        return self.debug_step


def make_add_bbox(
    click: tuple[float, float],
    image_width: int,
    image_height: int,
    bbox_size: int = 256,
) -> list[int]:
    """Build a clipped fixed bbox without importing the model runtime."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if bbox_size <= 0:
        raise ValueError("bbox_size must be greater than zero.")
    if not isinstance(click, (list, tuple)) or len(click) != 2:
        raise ValueError("click must be an [x, y] pair.")
    click_x, click_y = float(click[0]), float(click[1])
    if not math.isfinite(click_x) or not math.isfinite(click_y):
        raise ValueError("Click coordinates must be finite numbers.")
    if click_x < 0 or click_y < 0 or click_x >= image_width or click_y >= image_height:
        raise ValueError(
            f"Click ({click_x}, {click_y}) is outside image bounds "
            f"[0, {image_width - 1}] x [0, {image_height - 1}]."
        )

    box_width = min(int(bbox_size), image_width)
    box_height = min(int(bbox_size), image_height)
    x1 = min(max(int(round(click_x - box_width / 2.0)), 0), image_width - box_width)
    y1 = min(max(int(round(click_y - box_height / 2.0)), 0), image_height - box_height)
    return [x1, y1, x1 + box_width, y1 + box_height]


def _prompt_runtime(config_path: Path, checkpoint_path: Path, device: str):
    """Build and cache the existing Prompt predictor for repeated GUI clicks."""
    from tools.prompt_inference_utils import build_prompt_predictor, load_prompt_settings

    config = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    normalized_device = str(device).strip().lower()
    key = (str(config), str(checkpoint), normalized_device)
    if key not in _PROMPT_RUNTIME_CACHE:
        settings = load_prompt_settings(config, checkpoint=str(checkpoint))
        predictor, resolved_device = build_prompt_predictor(
            settings,
            device=normalized_device,
        )
        _PROMPT_RUNTIME_CACHE[key] = (settings, predictor, resolved_device)
    return _PROMPT_RUNTIME_CACHE[key]


def run_prompt_inference_on_tile(
    image_path: Path,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
    bbox: list[int],
    config_path: Path,
    checkpoint_path: Path,
    device: str,
    session_cache: PromptSessionCache | None = None,
    prompt_points: list[list[int]] | None = None,
    prompt_labels: list[int] | None = None,
    debug_prompt_points: bool = False,
    refine_by_points: bool = False,
    point_refine_mode: str = "distance",
    negative_margin: float = 0.0,
    negative_feature_refine: bool = False,
    negative_similarity_thr: float = 0.75,
    negative_feature_patch_size: int = 21,
    negative_protect_radius: int = 20,
    negative_spatial_sigma: float = 80.0,
    debug_refine_dir: Path | None = None,
    tile_id: str | None = None,
    session_id: str | None = None,
    click_order: list[dict] | None = None,
) -> dict | None:
    """Run existing SAMPoly Prompt inference and normalize its result contract."""
    if not positive_points:
        raise ValueError("At least one positive point is required for Prompt inference.")

    from tools.prompt_inference_utils import (
        infer_polygon_with_prompts,
        infer_polygon_with_prepared_prompts,
        load_rgb_image,
        normalize_bbox,
        polygon_from_prediction_mask,
        prepare_crop_prompts,
    )

    settings, predictor, resolved_device = _prompt_runtime(
        config_path,
        checkpoint_path,
        device,
    )
    image = (
        session_cache.get_image(image_path, load_rgb_image)
        if session_cache is not None else load_rgb_image(image_path)
    )
    image_height, image_width = image.shape[:2]
    normalized_bbox = normalize_bbox(bbox, image_width, image_height)
    if prompt_points is not None or prompt_labels is not None:
        if prompt_points is None or prompt_labels is None:
            raise ValueError("prompt_points and prompt_labels must be provided together.")
        if len(prompt_points) != len(prompt_labels):
            raise ValueError("prompt_points and prompt_labels must have the same length.")
        prompts = [
            (float(point[0]), float(point[1]), int(label))
            for point, label in zip(prompt_points, prompt_labels)
        ]
    else:
        prompts = [
            (float(x), float(y), 1) for x, y in positive_points
        ] + [
            (float(x), float(y), 0) for x, y in negative_points
        ]
    debug_paths = {}
    if session_cache is None:
        prediction = infer_polygon_with_prompts(
            predictor=predictor,
            image=image,
            bbox=normalized_bbox,
            prompts=prompts,
            include_bbox_center_prompt=True,
            crop_margin=0.075,
            multi_mask=bool(getattr(settings, "multi_mask", True)),
            max_distance=int(getattr(settings, "max_distance", 10)),
        )
    else:
        image_crop, crop_bbox, point_coords, point_labels, pos_transform = prepare_crop_prompts(
            image=image,
            bbox=normalized_bbox,
            prompts=prompts,
            include_bbox_center_prompt=True,
            crop_margin=0.075,
        )
        embedding_key = (
            str(session_cache.image_path),
            tuple(int(value) for value in normalized_bbox),
            tuple(int(value) for value in image_crop.shape[:2]),
            0.075,
        )
        session_cache.ensure_embedding(
            predictor,
            embedding_key,
            lambda: predictor.set_image_resize(image_crop),
        )
        prediction = infer_polygon_with_prepared_prompts(
            predictor=predictor,
            image=image,
            image_crop=image_crop,
            crop_bbox=crop_bbox,
            point_coords=point_coords,
            point_labels=point_labels,
            pos_transform=pos_transform,
            multi_mask=bool(getattr(settings, "multi_mask", True)),
            max_distance=int(getattr(settings, "max_distance", 10)),
        )
        before_point_refine_polygon = prediction["polygon"]
        positive_crop_points, negative_crop_points = split_labeled_points(
            prediction["point_coords"],
            prediction["point_labels"],
        )
        step_dir = None
        if debug_prompt_points or debug_refine_dir is not None:
            root_dir = Path(debug_refine_dir) if debug_refine_dir is not None else Path("work_dir") / "debug_refine"
            step_index = session_cache.next_debug_step()
            safe_tile_id = str(tile_id or Path(image_path).stem)
            safe_session_id = str(session_id or "session")
            step_dir = root_dir / safe_tile_id / safe_session_id / f"step_{step_index:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            coordinate_debug = []
            crop_x1 = int(round(float(pos_transform[0])))
            crop_y1 = int(round(float(pos_transform[1])))
            predictor_transform = getattr(getattr(predictor, "transform", None), "apply_coords", None)
            predictor_points = None
            if callable(predictor_transform):
                predictor_points = predictor_transform(
                    np.asarray(prediction["point_coords"], dtype=np.float32),
                    image_crop.shape[:2],
                )
            for index, (point, label_value) in enumerate(zip(prompts, [item[2] for item in prompts])):
                crop_xy = [
                    float(point[0]) - crop_x1,
                    float(point[1]) - crop_y1,
                ]
                predictor_xy = (
                    predictor_points[index + 1].tolist()
                    if predictor_points is not None and len(predictor_points) > index + 1
                    else None
                )
                coordinate_debug.append(
                    {
                        "raw_mouse_xy": [float(point[0]), float(point[1])],
                        "image_xy": [float(point[0]), float(point[1])],
                        "bbox_xyxy": [int(value) for value in normalized_bbox],
                        "crop_xy_if_used": crop_xy,
                        "predictor_xy_if_available": predictor_xy,
                        "label": int(label_value),
                    }
                )
            prompt_payload = {
                "tile_id": safe_tile_id,
                "session_id": safe_session_id,
                "step": step_index,
                "bbox": [int(value) for value in normalized_bbox],
                "points": [[float(point[0]), float(point[1])] for point in prompt_points or []],
                "labels": [int(label) for label in (prompt_labels or [])],
                "positive_points": [[float(x), float(y)] for x, y in positive_points],
                "negative_points": [[float(x), float(y)] for x, y in negative_points],
                "coordinate_debug": coordinate_debug,
            }
            debug_paths["prompt_points_json"] = str(
                save_prompt_points_json(step_dir / "prompt_points.json", prompt_payload)
            )
            if debug_prompt_points:
                print("[debug_prompt_points] bbox =", [int(value) for value in normalized_bbox])
                print("[debug_prompt_points] points =", prompt_payload["points"])
                print("[debug_prompt_points] labels =", prompt_payload["labels"])
                print("[debug_prompt_points] predictor_point_coords =", np.asarray(prediction["point_coords"]).tolist())
                print("[debug_prompt_points] predictor_point_labels =", np.asarray(prediction["point_labels"]).tolist())

        if refine_by_points:
            original_mask_before_point_refine = prediction["mask"].copy()
            if step_dir is not None:
                before_path = step_dir / "mask_before_point_refine.png"
                after_path = step_dir / "mask_after_point_refine.png"
                regions_path = step_dir / "mask_point_refine_regions.png"
                save_binary_mask(before_path, original_mask_before_point_refine[0])
                debug_paths["mask_before_point_refine"] = str(before_path)
            refined_mask, refined, refine_message = refine_mask_by_prompt_points(
                prediction["mask"],
                positive_crop_points,
                negative_crop_points,
                mode=point_refine_mode,
                negative_margin=negative_margin,
            )
            prediction["metadata"] = dict(prediction.get("metadata") or {})
            prediction["metadata"]["point_refine_message"] = refine_message
            if refined:
                crop_height, crop_width = image_crop.shape[:2]
                polygon_after_refine, polygon_score, extraction_metadata = _polygon_from_refined_mask(
                    refined_mask,
                    positive_crop_points,
                    pos_transform,
                    image.shape[1],
                    image.shape[0],
                )
                prediction["mask"] = refined_mask
                prediction["polygon"] = polygon_after_refine
                prediction["score"] = float(polygon_score)
                prediction["metadata"] = dict(prediction.get("metadata") or {})
                prediction["metadata"]["mask_polygon_extraction"] = extraction_metadata
            if step_dir is not None:
                save_binary_mask(after_path, prediction["mask"][0])
                save_point_refine_regions(
                    regions_path,
                    original_mask_before_point_refine[0],
                    prediction["mask"][0],
                    prediction["point_coords"],
                    prediction["point_labels"],
                )
                debug_paths["mask_after_point_refine"] = str(after_path)
                debug_paths["mask_point_refine_regions"] = str(regions_path)

        if negative_feature_refine:
            before_feature_mask = prediction["mask"][0].copy()
            before_polygon = prediction["polygon"]
            feature_refine = refine_mask_by_negative_features(
                image_crop,
                before_feature_mask,
                positive_crop_points,
                negative_crop_points,
                patch_size=negative_feature_patch_size,
                similarity_thr=negative_similarity_thr,
                protect_radius=negative_protect_radius,
                spatial_sigma=negative_spatial_sigma,
            )
            prediction["metadata"] = dict(prediction.get("metadata") or {})
            prediction["metadata"]["negative_feature_message"] = feature_refine["message"]
            if feature_refine["applied"]:
                crop_height, crop_width = image_crop.shape[:2]
                feature_mask, _, primary_message = keep_primary_component(
                    feature_refine["mask"].reshape(1, crop_height, crop_width),
                    positive_points=positive_crop_points,
                )
                polygon_after_feature, polygon_score, extraction_metadata = _polygon_from_refined_mask(
                    feature_mask,
                    positive_crop_points,
                    pos_transform,
                    image.shape[1],
                    image.shape[0],
                )
                prediction["mask"] = feature_mask
                prediction["polygon"] = polygon_after_feature
                prediction["score"] = float(polygon_score)
                prediction["metadata"] = dict(prediction.get("metadata") or {})
                prediction["metadata"]["negative_feature_primary_component"] = primary_message
                prediction["metadata"]["mask_polygon_extraction"] = extraction_metadata
            if step_dir is not None:
                save_negative_feature_debug(
                    str(step_dir),
                    before_feature_mask,
                    prediction["mask"][0],
                    feature_refine["similarity_map"],
                    feature_refine["prototypes"],
                    polygon_before=before_point_refine_polygon if refine_by_points else before_polygon,
                    polygon_after=prediction["polygon"],
                )
                for name in (
                    "negative_patch.png",
                    "negative_similarity_map.png",
                    "mask_before_refine.png",
                    "mask_after_refine.png",
                    "polygon_before_refine.png",
                    "polygon_after_refine.png",
                ):
                    debug_paths[name.replace(".png", "")] = str(step_dir / name)
    return {
        "success": True,
        "points_tile": [
            [float(x), float(y)] for x, y in prediction["polygon"]
        ],
        "mask": prediction.get("mask"),
        "score": float(prediction["score"]),
        "bbox_tile": [int(value) for value in normalized_bbox],
        "debug_paths": debug_paths,
        "metadata": {
            "model_score": float(prediction["model_score"]),
            "device": resolved_device,
            "positive_point_count": len(positive_points),
            "negative_point_count": len(negative_points),
            "mask_coordinate_space": "crop",
            "point_refine_enabled": bool(refine_by_points),
            "point_refine_mode": point_refine_mode,
            "negative_feature_refine": bool(negative_feature_refine),
            "click_order": click_order or [],
            **(prediction.get("metadata") or {}),
        },
    }


def run_mock_prompt_inference(
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
    bbox: list[int],
) -> dict:
    """Return a deterministic rectangle occupying 60 percent of the bbox."""
    if not positive_points:
        raise ValueError("At least one positive point is required for mock Prompt inference.")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2].")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height.")

    margin_scale = 0.2 + min(len(negative_points), 4) * 0.05
    margin_x = (x2 - x1) * margin_scale
    margin_y = (y2 - y1) * margin_scale
    mock_bbox = [x1 + margin_x, y1 + margin_y, x2 - margin_x, y2 - margin_y]
    points = [
        [mock_bbox[0], mock_bbox[1]],
        [mock_bbox[2], mock_bbox[1]],
        [mock_bbox[2], mock_bbox[3]],
        [mock_bbox[0], mock_bbox[3]],
    ]
    return {
        "success": True,
        "points_tile": points,
        "score": 0.0,
        "bbox_tile": mock_bbox,
        "debug_paths": {},
        "metadata": {
            "mock_prompt": True,
            "positive_point_count": len(positive_points),
            "negative_point_count": len(negative_points),
        },
    }


def _polygon_from_refined_mask(mask, positive_points, pos_transform, image_width, image_height):
    """Extract one robust polygon from a refined single-component mask."""
    from tools.prompt_inference_utils import validate_polygon
    from utils.post_process import transform_polygon_to_original

    polygon_crop, extraction_metadata = extract_main_polygon_from_mask(
        np.asarray(mask)[0],
        positive_points=positive_points,
    )
    if polygon_crop is None:
        raise ValueError(f"Failed to extract polygon from refined mask: {extraction_metadata}")
    polygon_original = transform_polygon_to_original(
        np.asarray(polygon_crop, dtype=np.float32),
        pos_transform,
    )
    polygon_validated = validate_polygon(
        polygon_original,
        image_width,
        image_height,
    )
    score = float(max(cv2.contourArea(np.asarray(polygon_crop, dtype=np.float32)), 1.0))
    return polygon_validated.tolist(), score, extraction_metadata
