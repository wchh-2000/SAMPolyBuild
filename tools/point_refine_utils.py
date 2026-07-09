"""Reusable point-guided mask refinement helpers for Prompt review and GUI tools."""

from pathlib import Path
import json

import cv2
import numpy as np


def keep_primary_component(
    mask,
    positive_points: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, bool, str]:
    """Keep one primary connected component, preferring components that contain positive points."""
    original_mask = np.asarray(mask).copy()
    if original_mask.ndim != 3 or original_mask.shape[0] != 1:
        raise ValueError(f"Expected mask shape (1, H, W), got {original_mask.shape}")
    binary_mask = (original_mask[0] > 0).astype(np.uint8)
    if np.count_nonzero(binary_mask) == 0:
        return original_mask, False, "Primary-component filtering received an empty mask."

    num_components, labels_map, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_components <= 2:
        return original_mask, False, "Mask already has one foreground component."

    candidate_components = []
    positive_components = set()
    for x, y in positive_points or []:
        if 0 <= x < binary_mask.shape[1] and 0 <= y < binary_mask.shape[0]:
            component_id = int(labels_map[y, x])
            if component_id > 0:
                positive_components.add(component_id)
    if positive_components:
        candidate_components = list(positive_components)
        selection_reason = "largest_positive_component"
    else:
        candidate_components = list(range(1, num_components))
        selection_reason = "largest_foreground_component"

    selected_component = max(
        candidate_components,
        key=lambda component_id: int(stats[component_id, cv2.CC_STAT_AREA]),
    )
    refined_mask = np.zeros_like(binary_mask, dtype=original_mask.dtype)
    refined_mask[labels_map == selected_component] = 1
    removed_components = [
        int(component_id) for component_id in range(1, num_components)
        if component_id != selected_component
    ]
    message = (
        "Primary-component filtering applied: "
        f"reason={selection_reason}, selected_component={int(selected_component)}, "
        f"selected_area={int(stats[selected_component, cv2.CC_STAT_AREA])}, "
        f"removed_components={removed_components}"
    )
    return refined_mask.reshape(1, refined_mask.shape[0], refined_mask.shape[1]), True, message


def extract_main_polygon_from_mask(
    mask_2d,
    positive_points: list[tuple[int, int]] | None = None,
    approx_epsilon_ratio: float = 0.01,
) -> tuple[list[list[float]] | None, dict]:
    """Extract one primary external contour from a binary mask and simplify it."""
    binary_mask = (np.asarray(mask_2d) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {
            "accepted": False,
            "reason": "no_external_contour",
            "contour_count": 0,
        }
    contour_infos = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        contains_positive = False
        for point in positive_points or []:
            if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
                contains_positive = True
                break
        contour_infos.append(
            {
                "contour": contour,
                "area": area,
                "contains_positive": contains_positive,
            }
        )
    if not contour_infos:
        return None, {
            "accepted": False,
            "reason": "all_external_contours_empty",
            "contour_count": int(len(contours)),
        }
    preferred = [item for item in contour_infos if item["contains_positive"]]
    pool = preferred or contour_infos
    selected = max(pool, key=lambda item: item["area"])
    perimeter = float(cv2.arcLength(selected["contour"], True))
    epsilon = max(perimeter * float(approx_epsilon_ratio), 1.0)
    approx = cv2.approxPolyDP(selected["contour"], epsilon, True)
    polygon = [
        [float(point[0][0]), float(point[0][1])]
        for point in approx
    ]
    return polygon, {
        "accepted": True,
        "reason": "positive_contour" if preferred else "largest_contour",
        "contour_count": int(len(contour_infos)),
        "selected_area": float(selected["area"]),
        "polygon_point_count": int(len(polygon)),
        "contains_positive": bool(selected["contains_positive"]),
    }


def split_prompt_points(
    point_coords,
    point_labels,
    width: int,
    height: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split crop-space prompt points into bounded positive and negative sets."""
    positive_points = []
    negative_points = []
    if point_coords is None or point_labels is None:
        return positive_points, negative_points
    for point, label in zip(point_coords, point_labels):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if not (0 <= x < width and 0 <= y < height):
            continue
        if int(label) == 1:
            positive_points.append((x, y))
        elif int(label) == 0:
            negative_points.append((x, y))
    return positive_points, negative_points


def split_labeled_points(
    point_coords,
    point_labels,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split any prompt arrays into positive and negative integer coordinates."""
    positive_points = []
    negative_points = []
    if point_coords is None or point_labels is None:
        return positive_points, negative_points
    for point, label_value in zip(point_coords, point_labels):
        item = (int(round(float(point[0]))), int(round(float(point[1]))))
        if int(label_value) == 1:
            positive_points.append(item)
        elif int(label_value) == 0:
            negative_points.append(item)
    return positive_points, negative_points


def save_binary_mask(path: Path, mask) -> None:
    """Save one binary mask image for debug inspection."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask_to_save = ((np.asarray(mask) > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(destination), mask_to_save)


def save_point_refine_regions(path: Path, original_mask, refined_mask, point_coords, point_labels) -> None:
    """Visualize kept and removed regions together with prompt points."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = (np.asarray(original_mask) > 0).astype(np.uint8)
    after = (np.asarray(refined_mask) > 0).astype(np.uint8)
    kept = (before == 1) & (after == 1)
    removed = (before == 1) & (after == 0)
    vis = np.zeros((before.shape[0], before.shape[1], 3), dtype=np.uint8)
    vis[before == 1] = (80, 80, 80)
    vis[kept] = (0, 180, 0)
    vis[removed] = (0, 0, 220)
    if point_coords is not None and point_labels is not None:
        for point, label in zip(point_coords, point_labels):
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            color = (0, 255, 0) if int(label) == 1 else (0, 0, 255)
            cv2.circle(vis, (x, y), 3, color, thickness=-1)
            cv2.circle(vis, (x, y), 5, (255, 255, 255), thickness=1)
    cv2.imwrite(str(destination), vis)


def refine_mask_by_connected_components(
    mask,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
) -> tuple[np.ndarray, bool, str]:
    """Remove negative components while keeping positive components when possible."""
    if np.asarray(mask).ndim != 3 or np.asarray(mask).shape[0] != 1:
        raise ValueError(f"Expected mask shape (1, H, W), got {np.asarray(mask).shape}")
    original_mask = np.asarray(mask).copy()
    binary_mask = (original_mask[0] > 0).astype(np.uint8)
    num_components, labels_map = cv2.connectedComponents(binary_mask, connectivity=8)
    all_components = set(range(1, num_components))
    positive_components = set()
    negative_components = set()

    for x, y in positive_points:
        if 0 <= x < binary_mask.shape[1] and 0 <= y < binary_mask.shape[0]:
            component_id = int(labels_map[y, x])
            if component_id > 0:
                positive_components.add(component_id)
    for x, y in negative_points:
        if 0 <= x < binary_mask.shape[1] and 0 <= y < binary_mask.shape[0]:
            component_id = int(labels_map[y, x])
            if component_id > 0:
                negative_components.add(component_id)

    if positive_components:
        keep_components = {
            max(
                positive_components,
                key=lambda component_id: int(np.count_nonzero(labels_map == component_id)),
            )
        }
    elif negative_components:
        keep_components = all_components - negative_components
    else:
        return original_mask, False, "No negative components were hit; using original mask."

    if not keep_components:
        return original_mask, False, "Point refinement would remove all components; using original mask."

    refined_mask = np.zeros_like(binary_mask, dtype=original_mask.dtype)
    for component_id in keep_components:
        refined_mask[labels_map == component_id] = 1
    if np.count_nonzero(refined_mask) == 0:
        return original_mask, False, "Point refinement removed all pixels; using original mask."

    refined_mask_3d = refined_mask.reshape(1, refined_mask.shape[0], refined_mask.shape[1])
    refined_mask_3d, primary_applied, primary_message = keep_primary_component(
        refined_mask_3d,
        positive_points=positive_points,
    )
    message = (
        "Connected-component point refinement applied: "
        f"positive_components={sorted(positive_components)}, "
        f"negative_components={sorted(negative_components)}, "
        f"kept_components={sorted(keep_components)}"
    )
    if primary_applied:
        message = f"{message}; {primary_message}"
    return refined_mask_3d, True, message


def refine_mask_by_distance(
    mask,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
    negative_margin: float = 0.0,
) -> tuple[np.ndarray, bool, str]:
    """Keep mask pixels closer to positives than negatives, with positive protection."""
    if np.asarray(mask).ndim != 3 or np.asarray(mask).shape[0] != 1:
        raise ValueError(f"Expected mask shape (1, H, W), got {np.asarray(mask).shape}")
    original_mask = np.asarray(mask).copy()
    binary_mask = (original_mask[0] > 0).astype(np.uint8)
    if not positive_points or not negative_points:
        return original_mask, False, (
            "Distance point refinement requires at least one positive and one negative point; "
            "using original mask."
        )

    ys, xs = np.where(binary_mask > 0)
    if xs.size == 0:
        return original_mask, False, "Distance point refinement received an empty mask; using original mask."

    pixels = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    pos = np.asarray(positive_points, dtype=np.float32)
    neg = np.asarray(negative_points, dtype=np.float32)
    dist_to_pos = np.sqrt(((pixels[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    dist_to_neg = np.sqrt(((pixels[:, None, :] - neg[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    keep_pixels = dist_to_pos + float(negative_margin) < dist_to_neg

    refined_mask_2d = np.zeros_like(binary_mask, dtype=original_mask.dtype)
    refined_mask_2d[ys[keep_pixels], xs[keep_pixels]] = 1
    for x, y in positive_points:
        if 0 <= x < refined_mask_2d.shape[1] and 0 <= y < refined_mask_2d.shape[0]:
            refined_mask_2d[y, x] = 1
    if np.count_nonzero(refined_mask_2d) == 0:
        return original_mask, False, "Distance point refinement removed all mask pixels; using original mask."

    removed_pixels = int(xs.size - np.count_nonzero(refined_mask_2d))
    message = (
        "Distance point refinement applied: "
        f"positive_points={positive_points}, negative_points={negative_points}, "
        f"negative_margin={negative_margin}, removed_pixels={removed_pixels}, "
        f"kept_pixels={int(np.count_nonzero(refined_mask_2d))}"
    )
    refined_mask_3d, primary_applied, primary_message = keep_primary_component(
        refined_mask_2d.reshape(1, refined_mask_2d.shape[0], refined_mask_2d.shape[1]),
        positive_points=positive_points,
    )
    if primary_applied:
        message = f"{message}; {primary_message}"
    return refined_mask_3d, True, message


def refine_mask_by_prompt_points(
    mask,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
    mode: str = "distance",
    negative_margin: float = 0.0,
) -> tuple[np.ndarray, bool, str]:
    """Refine one mask from split prompt point sets using the requested mode."""
    if mode == "connected_component":
        return refine_mask_by_connected_components(mask, positive_points, negative_points)
    if mode == "distance":
        return refine_mask_by_distance(mask, positive_points, negative_points, negative_margin)
    raise ValueError(f"Unsupported point_refine_mode: {mode}")


def save_prompt_points_json(path: Path, payload: dict) -> Path:
    """Save one prompt debug payload with stable UTF-8 JSON formatting."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination
