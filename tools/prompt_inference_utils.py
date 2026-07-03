import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from utils.auto_result_bbox import infer_image_id, load_and_select_bbox
from utils.post_process import GetPolygons, transform_polygon_to_original


DEFAULT_PROMPT_CHECKPOINT = "prompt_interactive.pth"
DEFAULT_PROMPT_CONFIG = "configs/prompt_instance_spacenet.json"


def load_prompt_settings(config_path, checkpoint=None):
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Prompt config not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt config must contain a JSON object: {config_file}")

    settings = dict(data)
    settings["checkpoint"] = checkpoint or DEFAULT_PROMPT_CHECKPOINT
    return SimpleNamespace(**settings)


def build_prompt_predictor(settings, gpu=0, device=None):
    import torch
    from segment_anything import SamPredictor, build_sam

    checkpoint = Path(settings.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Prompt checkpoint not found: {checkpoint}")

    if device is not None:
        resolved_device = str(device)
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {resolved_device}")
    elif gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a CUDA-enabled environment or pass --gpu -1 for CPU.")
        resolved_device = f"cuda:{gpu}"
    else:
        resolved_device = "cpu"

    model_args = vars(settings).copy()
    model_args["checkpoint"] = str(checkpoint)
    sam_model = build_sam(use_poly=True, load_pl=True, **model_args).to(resolved_device)
    sam_model.eval()
    return SamPredictor(sam_model, polygon=True), resolved_device


def load_rgb_image(image_path):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to read input image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def validate_click(click, image_width, image_height):
    x, y = float(click[0]), float(click[1])
    if not np.isfinite([x, y]).all():
        raise ValueError("Click coordinates must be finite numbers.")
    if x < 0 or y < 0 or x >= image_width or y >= image_height:
        raise ValueError(
            f"Click ({x}, {y}) is outside image bounds "
            f"[0, {image_width - 1}] x [0, {image_height - 1}]."
        )
    return x, y


def make_fixed_bbox(click, image_width, image_height, bbox_size):
    if bbox_size <= 0:
        raise ValueError("bbox_size must be greater than zero.")
    click_x, click_y = validate_click(click, image_width, image_height)
    box_width = min(int(bbox_size), image_width)
    box_height = min(int(bbox_size), image_height)

    x1 = int(round(click_x - box_width / 2.0))
    y1 = int(round(click_y - box_height / 2.0))
    x1 = min(max(x1, 0), image_width - box_width)
    y1 = min(max(y1, 0), image_height - box_height)
    return [x1, y1, x1 + box_width, y1 + box_height]


def select_bbox(
    mode,
    click,
    image_width,
    image_height,
    bbox_size,
    auto_results=None,
    auto_image_id=None,
    auto_min_score=0.0,
    image_path=None,
):
    fixed_bbox = make_fixed_bbox(click, image_width, image_height, bbox_size)
    if mode == "fixed":
        return fixed_bbox, "fixed", {"fallback_used": False}
    if mode != "auto":
        raise ValueError(f"Unsupported bbox_mode: {mode}")
    if not auto_results:
        return fixed_bbox, "fixed", {
            "fallback_used": True,
            "fallback_reason": "--auto_results was not provided.",
        }

    image_id = auto_image_id or infer_image_id(image_path)
    try:
        bbox, info = load_and_select_bbox(
            auto_results,
            click,
            image_id=image_id,
            min_score=auto_min_score,
        )
        if not info.get("contains_click", False):
            return fixed_bbox, "fixed", {
                "fallback_used": True,
                "fallback_reason": "No auto bbox contains the click point.",
                "auto_candidate": info,
            }
        normalized = normalize_bbox(bbox, image_width, image_height)
        return normalized, "auto", info
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return fixed_bbox, "fixed", {
            "fallback_used": True,
            "fallback_reason": str(exc),
        }


def normalize_bbox(bbox, image_width, image_height):
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must contain four XYXY values.")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = int(max(0, min(round(x1), image_width - 1)))
    y1 = int(max(0, min(round(y1), image_height - 1)))
    x2 = int(max(1, min(round(x2), image_width)))
    y2 = int(max(1, min(round(y2), image_height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox has no valid area after clipping: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]


def prepare_crop_prompt(image, bbox, click, crop_margin=0.075):
    return prepare_crop_prompts(
        image=image,
        bbox=bbox,
        prompts=[(float(click[0]), float(click[1]), 1)],
        include_bbox_center_prompt=True,
        crop_margin=crop_margin,
    )


def prepare_crop_prompts(
    image,
    bbox,
    prompts,
    include_bbox_center_prompt=True,
    crop_margin=0.1,
):
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = normalize_bbox(bbox, image_width, image_height)
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    crop_x1 = max(0, int(x1 - bbox_width * crop_margin))
    crop_y1 = max(0, int(y1 - bbox_height * crop_margin))
    crop_x2 = min(image_width, int(x2 + bbox_width * crop_margin))
    crop_y2 = min(image_height, int(y2 + bbox_height * crop_margin))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("Expanded bbox produced an empty image crop.")

    image_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_bbox = np.asarray(
        [x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1],
        dtype=np.float32,
    )
    bbox_center = np.asarray(
        [(crop_bbox[0] + crop_bbox[2]) / 2.0, (crop_bbox[1] + crop_bbox[3]) / 2.0],
        dtype=np.float32,
    )
    normalized_prompts = normalize_prompts(prompts, image_width, image_height)
    point_coords = [
        np.asarray([x - crop_x1, y - crop_y1], dtype=np.float32)
        for x, y, _ in normalized_prompts
    ]
    point_labels = [label for _, _, label in normalized_prompts]
    if include_bbox_center_prompt:
        point_coords.insert(0, bbox_center)
        point_labels.insert(0, 1)

    point_coords = np.stack(point_coords, axis=0).astype(np.float32)
    point_labels = np.asarray(point_labels, dtype=np.int32)
    pos_transform = [crop_x1, crop_y1, 1, 1]
    return image_crop, crop_bbox, point_coords, point_labels, pos_transform


def normalize_prompts(prompts, image_width, image_height):
    if not prompts:
        raise ValueError("At least one prompt point is required.")
    normalized = []
    for prompt in prompts:
        if hasattr(prompt, "x") and hasattr(prompt, "y") and hasattr(prompt, "label"):
            x, y, label = prompt.x, prompt.y, prompt.label
        elif isinstance(prompt, (list, tuple)) and len(prompt) == 3:
            x, y, label = prompt
        else:
            raise ValueError("Each prompt must provide x, y, and label values.")
        x, y = validate_click((x, y), image_width, image_height)
        label_value = float(label)
        if not np.isfinite(label_value) or label_value not in (0.0, 1.0):
            raise ValueError("Prompt label must be 0 (negative) or 1 (positive).")
        label = int(label_value)
        normalized.append((x, y, label))
    if not any(label == 1 for _, _, label in normalized):
        raise ValueError("At least one positive prompt point is required.")
    return normalized


def infer_polygon_with_prompts(
    predictor,
    image,
    bbox,
    prompts,
    include_bbox_center_prompt=True,
    crop_margin=0.1,
    multi_mask=True,
    max_distance=10,
):
    import torch

    image_crop, crop_bbox, point_coords, point_labels, pos_transform = prepare_crop_prompts(
        image,
        bbox,
        prompts,
        include_bbox_center_prompt=include_bbox_center_prompt,
        crop_margin=crop_margin,
    )
    predictor.set_image_resize(image_crop)
    masks, model_scores, _, pred_poly = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=crop_bbox,
        multimask_output=multi_mask,
    )

    crop_height, crop_width = image_crop.shape[:2]
    selected_index = int(np.argmax(model_scores)) if multi_mask else 0
    selected_mask = masks[selected_index].reshape(1, crop_height, crop_width)
    pred_vmap = torch.sigmoid(pred_poly["vmap"])
    pred_voff = torch.sigmoid(pred_poly["voff"])
    polygons, polygon_scores, valid_mask = GetPolygons(
        selected_mask,
        pred_vmap,
        pred_voff,
        ori_size=(crop_width, crop_height),
        max_distance=max_distance,
    )
    if not valid_mask[0] or polygons[0] is None:
        raise ValueError("SAMPoly did not produce a valid polygon for the selected mask.")

    polygon = transform_polygon_to_original(polygons[0], pos_transform)
    polygon = validate_polygon(polygon, image.shape[1], image.shape[0])
    return {
        "polygon": polygon,
        "score": float(polygon_scores[0]),
        "model_score": float(model_scores[selected_index]),
        "mask": selected_mask,
        "point_coords": point_coords,
        "point_labels": point_labels,
    }


def infer_single_polygon(predictor, image, bbox, click, multi_mask=True, max_distance=10):
    return infer_polygon_with_prompts(
        predictor=predictor,
        image=image,
        bbox=bbox,
        prompts=[(float(click[0]), float(click[1]), 1)],
        include_bbox_center_prompt=True,
        crop_margin=0.075,
        multi_mask=multi_mask,
        max_distance=max_distance,
    )


def validate_polygon(polygon, image_width, image_height):
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError(f"Polygon must have shape (N, 2), got {polygon.shape}.")
    if polygon.shape[0] < 3:
        raise ValueError(f"Polygon must contain at least 3 points, got {polygon.shape[0]}.")
    if not np.isfinite(polygon).all():
        raise ValueError("Polygon contains non-finite coordinates.")

    polygon[:, 0] = np.clip(polygon[:, 0], 0, image_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, image_height - 1)
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    area = 0.5 * abs(
        np.dot(polygon[:-1, 0], polygon[1:, 1])
        - np.dot(polygon[:-1, 1], polygon[1:, 0])
    )
    if area <= 0:
        raise ValueError("Polygon area must be greater than zero.")
    return polygon
