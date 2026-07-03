from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class AutoBBoxConfig:
    init_window_size: int = 96
    min_box_size: int = 12
    max_box_size: int = 256
    padding: int = 8
    floodfill_lo_diff: int = 12
    floodfill_up_diff: int = 12
    canny_low: int = 50
    canny_high: int = 150
    morph_kernel_size: int = 3
    fallback_box_size: int = 64
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.8
    max_aspect_ratio: float = 6.0


def generate_adaptive_bbox(image, click_point, config=None, debug=False):
    """
    image: np.ndarray, supports gray/RGB/BGR/RGBA
    click_point: (x, y)
    config: dict or AutoBBoxConfig
    debug: bool

    return:
        bbox: [x1, y1, x2, y2]
        info: dict
    """
    cfg = AutoBBoxConfig()
    info = _base_info(click_point, debug)
    try:
        cfg = _normalize_config(config)
        info["config"] = asdict(cfg)
        valid_image, message = _validate_image(image)
        if not valid_image:
            bbox = _empty_image_fallback(click_point, cfg)
            info.update(
                success=False,
                method="fallback",
                message=message,
                final_bbox=bbox,
                fallback_used=True,
            )
            return bbox, info

        height, width = image.shape[:2]
        point, _ = _normalize_point(click_point)
        if point is None:
            info.update(
                success=False,
                method="fallback",
                message="Invalid click_point; expected numeric (x, y).",
                fallback_used=True,
            )
            bbox = _fallback_bbox((0, 0), width, height, cfg)
            info["final_bbox"] = bbox
            return bbox, info

        x, y = point
        info["click_point"] = [x, y]
        click_inside = 0 <= x < width and 0 <= y < height
        if not click_inside:
            clamped_point = (int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1)))
            bbox = _fallback_bbox(clamped_point, width, height, cfg)
            info.update(
                success=False,
                method="fallback",
                message="click_point is outside image bounds; returned clipped fallback bbox.",
                final_bbox=bbox,
                fallback_used=True,
            )
            return bbox, info

        roi, roi_origin, roi_box = _crop_roi(image, point, cfg.init_window_size)
        info["roi"] = roi_box
        gray, gray_message = _to_gray(roi)
        if gray is None:
            bbox = _fallback_bbox(point, width, height, cfg)
            info.update(
                success=False,
                method="fallback",
                message=gray_message,
                final_bbox=bbox,
                fallback_used=True,
            )
            return bbox, info

        mask = _floodfill_mask(gray, (x - roi_origin[0], y - roi_origin[1]), cfg)
        mask = _cleanup_mask(mask, cfg.morph_kernel_size)
        raw_bbox, mask_area, mask_message = _component_bbox_containing_point(
            mask, (x - roi_origin[0], y - roi_origin[1])
        )
        info["mask_area"] = int(mask_area)
        if debug:
            info["debug_mask"] = mask.copy()

        if raw_bbox is None:
            bbox = _fallback_bbox(point, width, height, cfg)
            info.update(
                success=False,
                method="fallback",
                message=mask_message,
                final_bbox=bbox,
                fallback_used=True,
            )
            return bbox, info

        info["raw_bbox"] = _bbox_to_list(raw_bbox)
        final_bbox = _map_bbox_to_image(raw_bbox, roi_origin, cfg.padding, width, height)
        valid, valid_message = _validate_bbox(final_bbox, point, width, height, mask_area, mask.size, cfg)
        if not valid:
            bbox = _fallback_bbox(point, width, height, cfg)
            info.update(
                success=False,
                method="fallback",
                message=valid_message,
                final_bbox=bbox,
                fallback_used=True,
            )
            return bbox, info

        info.update(
            success=True,
            method="floodfill",
            message="adaptive bbox generated.",
            final_bbox=final_bbox,
            fallback_used=False,
        )
        return final_bbox, info
    except Exception as exc:
        bbox = _safe_fallback_from_image(image, click_point, cfg)
        info.update(
            success=False,
            method="fallback",
            message=f"adaptive bbox failed: {exc}",
            final_bbox=bbox,
            fallback_used=True,
        )
        return bbox, info


def _normalize_config(config: Optional[Any]) -> AutoBBoxConfig:
    if config is None:
        return AutoBBoxConfig()
    if isinstance(config, AutoBBoxConfig):
        return config
    if isinstance(config, dict):
        values = asdict(AutoBBoxConfig())
        values.update(config)
        return AutoBBoxConfig(**values)
    raise TypeError("config must be None, dict, or AutoBBoxConfig.")


def _base_info(click_point, debug: bool) -> Dict[str, Any]:
    return {
        "success": False,
        "method": None,
        "message": "",
        "click_point": list(click_point) if _is_point_like(click_point) else click_point,
        "roi": None,
        "raw_bbox": None,
        "final_bbox": None,
        "mask_area": 0,
        "fallback_used": False,
        "debug_mask": None if debug else "disabled",
    }


def _validate_image(image) -> Tuple[bool, str]:
    if image is None:
        return False, "image is None."
    if not isinstance(image, np.ndarray):
        return False, "image must be a numpy ndarray."
    if image.size == 0:
        return False, "image is empty."
    if image.ndim not in (2, 3):
        return False, "image must be gray or HWC color array."
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        return False, "image has invalid spatial shape."
    return True, ""


def _is_point_like(point) -> bool:
    return isinstance(point, (tuple, list, np.ndarray)) and len(point) == 2


def _normalize_point(point) -> Tuple[Optional[Tuple[int, int]], bool]:
    if not _is_point_like(point):
        return None, False
    try:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
    except (TypeError, ValueError):
        return None, False
    return (x, y), True


def _crop_roi(image: np.ndarray, point: Tuple[int, int], window_size: int):
    height, width = image.shape[:2]
    half = max(1, int(window_size) // 2)
    x, y = point
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(width, x + half)
    y2 = min(height, y + half)
    return image[y1:y2, x1:x2], (x1, y1), [x1, y1, x2, y2]


def _to_gray(roi: np.ndarray):
    if roi.ndim == 2:
        return _as_uint8(roi), ""
    if roi.ndim != 3:
        return None, "ROI channel layout is invalid."
    channels = roi.shape[2]
    if channels == 1:
        return _as_uint8(roi[:, :, 0]), ""
    if channels == 3:
        return _as_uint8(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)), ""
    if channels == 4:
        return _as_uint8(cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)), ""
    return None, f"Unsupported image channel count: {channels}."


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        finite = np.nan_to_num(image, copy=True)
        if finite.max(initial=0) <= 1.0:
            finite = finite * 255.0
        return np.clip(finite, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def _floodfill_mask(gray: np.ndarray, seed: Tuple[int, int], cfg: AutoBBoxConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    flood_mask = np.zeros((enhanced.shape[0] + 2, enhanced.shape[1] + 2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(
        enhanced.copy(),
        flood_mask,
        seedPoint=seed,
        newVal=0,
        loDiff=int(cfg.floodfill_lo_diff),
        upDiff=int(cfg.floodfill_up_diff),
        flags=flags,
    )
    region = flood_mask[1:-1, 1:-1]
    edges = cv2.Canny(enhanced, int(cfg.canny_low), int(cfg.canny_high))
    return np.where((region > 0) & (edges == 0), 255, 0).astype(np.uint8)


def _cleanup_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    size = max(1, int(kernel_size))
    kernel = np.ones((size, size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _component_bbox_containing_point(mask: np.ndarray, point: Tuple[int, int]):
    x, y = point
    if mask.size == 0 or not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        return None, 0, "click point is outside ROI mask."
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    label_id = labels[y, x]
    if num_labels <= 1 or label_id == 0:
        return None, 0, "floodFill did not produce a valid component containing click_point."
    left, top, width, height, area = stats[label_id]
    return (int(left), int(top), int(left + width), int(top + height)), int(area), "component found."


def _map_bbox_to_image(raw_bbox, roi_origin, padding: int, image_width: int, image_height: int):
    x1, y1, x2, y2 = raw_bbox
    ox, oy = roi_origin
    pad = max(0, int(padding))
    mapped = [
        int(np.clip(x1 + ox - pad, 0, image_width - 1)),
        int(np.clip(y1 + oy - pad, 0, image_height - 1)),
        int(np.clip(x2 + ox + pad, 1, image_width)),
        int(np.clip(y2 + oy + pad, 1, image_height)),
    ]
    return mapped


def _validate_bbox(
    bbox,
    point: Tuple[int, int],
    image_width: int,
    image_height: int,
    mask_area: int,
    roi_area: int,
    cfg: AutoBBoxConfig,
):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False, "bbox has non-positive width or height."
    width = x2 - x1
    height = y2 - y1
    if width < cfg.min_box_size or height < cfg.min_box_size:
        return False, "bbox is smaller than min_box_size."
    if width > cfg.max_box_size or height > cfg.max_box_size:
        return False, "bbox is larger than max_box_size."
    aspect = max(width / max(height, 1), height / max(width, 1))
    if aspect > cfg.max_aspect_ratio:
        return False, "bbox aspect ratio is abnormal."
    x, y = point
    if not (x1 <= x < x2 and y1 <= y < y2):
        return False, "bbox does not contain click_point."
    area_ratio = float(mask_area) / float(max(roi_area, 1))
    if area_ratio < cfg.min_area_ratio:
        return False, "component area is too small."
    if area_ratio > cfg.max_area_ratio:
        return False, "component area is too large."
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return False, "bbox is outside image bounds."
    return True, "bbox is valid."


def _fallback_bbox(point: Tuple[int, int], image_width: int, image_height: int, cfg: AutoBBoxConfig):
    x, y = point
    half = max(1, int(cfg.fallback_box_size) // 2)
    x1 = int(np.clip(x - half, 0, max(0, image_width - 1)))
    y1 = int(np.clip(y - half, 0, max(0, image_height - 1)))
    x2 = int(np.clip(x + half, min(1, image_width), image_width))
    y2 = int(np.clip(y + half, min(1, image_height), image_height))
    if x2 <= x1:
        x2 = min(image_width, x1 + 1)
    if y2 <= y1:
        y2 = min(image_height, y1 + 1)
    return [x1, y1, x2, y2]


def _empty_image_fallback(click_point, cfg: AutoBBoxConfig):
    point, _ = _normalize_point(click_point)
    x, y = point if point is not None else (0, 0)
    size = max(1, int(cfg.fallback_box_size))
    return [x, y, x + size, y + size]


def _safe_fallback_from_image(image, click_point, cfg: AutoBBoxConfig):
    point, _ = _normalize_point(click_point)
    if point is None:
        point = (0, 0)
    if isinstance(image, np.ndarray) and image.ndim >= 2 and image.shape[0] > 0 and image.shape[1] > 0:
        height, width = image.shape[:2]
        point = (int(np.clip(point[0], 0, width - 1)), int(np.clip(point[1], 0, height - 1)))
        return _fallback_bbox(point, width, height, cfg)
    return _empty_image_fallback(point, cfg)


def _bbox_to_list(bbox):
    return [int(v) for v in bbox]
