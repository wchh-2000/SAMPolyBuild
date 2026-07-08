"""Thin Prompt inference adapter for tile-review click additions."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_PROMPT_RUNTIME_CACHE = {}


@dataclass
class PromptSessionCache:
    """Cache one tile image and its active crop embedding during Refine."""

    image_path: Path | None = None
    image: Any = None
    embedding_key: tuple | None = None
    _embedding_owner: int | None = None
    prepare_count: int = 0

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
) -> dict | None:
    """Run existing SAMPoly Prompt inference and normalize its result contract."""
    if not positive_points:
        raise ValueError("At least one positive point is required for Prompt inference.")

    from tools.prompt_inference_utils import (
        infer_polygon_with_prompts,
        infer_polygon_with_prepared_prompts,
        load_rgb_image,
        normalize_bbox,
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
    prompts = [
        (float(x), float(y), 1) for x, y in positive_points
    ] + [
        (float(x), float(y), 0) for x, y in negative_points
    ]
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
    return {
        "points_tile": [
            [float(x), float(y)] for x, y in prediction["polygon"]
        ],
        "score": float(prediction["score"]),
        "bbox_tile": [int(value) for value in normalized_bbox],
        "metadata": {
            "model_score": float(prediction["model_score"]),
            "device": resolved_device,
            "positive_point_count": len(positive_points),
            "negative_point_count": len(negative_points),
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

    margin_x = (x2 - x1) * 0.2
    margin_y = (y2 - y1) * 0.2
    mock_bbox = [x1 + margin_x, y1 + margin_y, x2 - margin_x, y2 - margin_y]
    points = [
        [mock_bbox[0], mock_bbox[1]],
        [mock_bbox[2], mock_bbox[1]],
        [mock_bbox[2], mock_bbox[3]],
        [mock_bbox[0], mock_bbox[3]],
    ]
    return {
        "points_tile": points,
        "score": 0.0,
        "bbox_tile": mock_bbox,
        "metadata": {
            "mock_prompt": True,
            "positive_point_count": len(positive_points),
            "negative_point_count": len(negative_points),
        },
    }
