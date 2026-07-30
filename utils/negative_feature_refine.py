import cv2
import numpy as np


class NegativeFeaturePrototype:
    def __init__(self, patch_size=21):
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if patch_size % 2 == 0:
            patch_size += 1
        self.patch_size = patch_size

    def extract(self, image, negative_point):
        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty ndarray.")
        x, y = _normalize_point(negative_point)
        patch = _crop_patch(image, x, y, self.patch_size)
        if patch.size == 0:
            raise ValueError(f"negative_point {negative_point} produced an empty patch.")
        return {
            "vector": _feature_vector(patch),
            "patch": patch.copy(),
            "point": (x, y),
        }


def feature_distance(pixel_features, negative_feature):
    vector = negative_feature["vector"] if isinstance(negative_feature, dict) else negative_feature
    vector = np.asarray(vector, dtype=np.float32)
    mean = np.concatenate([vector[0:3], vector[6:9], vector[12:15]], axis=0)
    std = np.concatenate([vector[3:6], vector[9:12], vector[15:18]], axis=0)
    std = np.maximum(std, 8.0)
    normalized = (pixel_features - mean.reshape(1, -1)) / std.reshape(1, -1)
    return np.sqrt((normalized ** 2).sum(axis=1))


def refine_mask_by_negative_features(
    image,
    mask,
    positive_points,
    negative_points,
    patch_size=21,
    similarity_thr=0.75,
    protect_radius=20,
    spatial_sigma=80,
):
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}.")
    binary_mask = (mask > 0).astype(np.uint8)
    if np.count_nonzero(binary_mask) == 0:
        return _result(mask, mask, None, None, [], False, "Mask is empty; skipped feature refinement.")

    pos_points = [_normalize_point(p) for p in positive_points]
    neg_points = [_normalize_point(p) for p in negative_points]
    if not neg_points:
        return _result(mask, mask, None, None, [], False, "No negative points; skipped feature refinement.")

    prototype_builder = NegativeFeaturePrototype(patch_size=patch_size)
    prototypes = [prototype_builder.extract(image, point) for point in neg_points]

    pixel_features, ys, xs = _features_for_mask_pixels(image, binary_mask)
    distances = _nearest_feature_distance(pixel_features, prototypes)
    feature_similarity = _distance_to_similarity(distances)
    spatial_weight = _spatial_weight(xs, ys, neg_points, spatial_sigma)
    score = feature_similarity * spatial_weight

    remove_pixels = score > float(similarity_thr)
    protect_mask = _positive_protection_mask(mask.shape, pos_points, protect_radius)
    if np.any(protect_mask):
        protected = protect_mask[ys, xs]
        remove_pixels = remove_pixels & (~protected)

    refined = binary_mask.copy()
    refined[ys[remove_pixels], xs[remove_pixels]] = 0
    if np.count_nonzero(refined) == 0:
        return _result(
            mask,
            mask,
            _scatter_to_map(mask.shape, ys, xs, score),
            _scatter_to_map(mask.shape, ys, xs, remove_pixels.astype(np.float32)),
            prototypes,
            False,
            "Negative feature refinement removed all mask pixels; using original mask.",
        )

    similarity_map = _scatter_to_map(mask.shape, ys, xs, score)
    remove_map = _scatter_to_map(mask.shape, ys, xs, remove_pixels.astype(np.float32))
    removed_pixels = int(np.count_nonzero(binary_mask) - np.count_nonzero(refined))
    message = (
        "Negative feature refinement applied: "
        f"negative_points={neg_points}, positive_points={pos_points}, "
        f"similarity_thr={similarity_thr}, removed_pixels={removed_pixels}, "
        f"kept_pixels={int(np.count_nonzero(refined))}"
    )
    return _result(mask, refined.astype(mask.dtype), similarity_map, remove_map, prototypes, True, message)


def save_negative_feature_debug(
    out_dir,
    before_mask,
    after_mask,
    similarity_map,
    prototypes,
    polygon_before=None,
    polygon_after=None,
):
    cv2.imwrite(f"{out_dir}/mask_before_refine.png", _to_uint8_mask(before_mask))
    cv2.imwrite(f"{out_dir}/mask_after_refine.png", _to_uint8_mask(after_mask))
    if similarity_map is not None:
        cv2.imwrite(f"{out_dir}/negative_similarity_map.png", _to_heatmap(similarity_map))
    if prototypes:
        cv2.imwrite(f"{out_dir}/negative_patch.png", _ensure_bgr(prototypes[0]["patch"]))
    if polygon_before is not None:
        cv2.imwrite(f"{out_dir}/polygon_before_refine.png", _polygon_canvas(before_mask.shape, polygon_before))
    if polygon_after is not None:
        cv2.imwrite(f"{out_dir}/polygon_after_refine.png", _polygon_canvas(after_mask.shape, polygon_after))


def _result(original, refined, similarity_map, remove_map, prototypes, applied, message):
    return {
        "mask": refined,
        "original_mask": original,
        "similarity_map": similarity_map,
        "remove_map": remove_map,
        "prototypes": prototypes,
        "applied": applied,
        "message": message,
    }


def _normalize_point(point):
    if point is None or len(point) != 2:
        raise ValueError("point must be an (x, y) pair.")
    return int(round(float(point[0]))), int(round(float(point[1])))


def _crop_patch(image, x, y, patch_size):
    height, width = image.shape[:2]
    half = patch_size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(width, x + half + 1)
    y2 = min(height, y + half + 1)
    return image[y1:y2, x1:x2]


def _feature_vector(patch):
    patch = _ensure_rgb(patch)
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    vectors = []
    for space in (patch, hsv, lab):
        values = space.reshape(-1, 3).astype(np.float32)
        vectors.extend(values.mean(axis=0).tolist())
        vectors.extend(values.std(axis=0).tolist())
    return np.asarray(vectors, dtype=np.float32)


def _features_for_mask_pixels(image, mask):
    rgb = _ensure_rgb(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    ys, xs = np.where(mask > 0)
    features = np.concatenate(
        [
            rgb[ys, xs].astype(np.float32),
            hsv[ys, xs].astype(np.float32),
            lab[ys, xs].astype(np.float32),
        ],
        axis=1,
    )
    return features, ys, xs


def _nearest_feature_distance(pixel_features, prototypes):
    distances = np.stack([feature_distance(pixel_features, prototype) for prototype in prototypes], axis=1)
    return distances.min(axis=1)


def _distance_to_similarity(distances):
    if distances.size == 0:
        return distances
    scale = np.percentile(distances, 90)
    if scale <= 1e-6:
        scale = float(distances.max()) if distances.max() > 1e-6 else 1.0
    similarity = 1.0 - np.clip(distances / scale, 0.0, 1.0)
    return similarity.astype(np.float32)


def _spatial_weight(xs, ys, negative_points, spatial_sigma):
    if spatial_sigma is None or spatial_sigma <= 0:
        return np.ones_like(xs, dtype=np.float32)
    points = np.asarray(negative_points, dtype=np.float32)
    pixels = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dist = np.sqrt(((pixels[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return np.exp(-(dist ** 2) / (2.0 * float(spatial_sigma) ** 2)).astype(np.float32)


def _positive_protection_mask(shape, positive_points, protect_radius):
    mask = np.zeros(shape, dtype=np.uint8)
    if protect_radius is None or protect_radius <= 0:
        return mask.astype(bool)
    for x, y in positive_points:
        cv2.circle(mask, (x, y), int(protect_radius), 1, thickness=-1)
    return mask.astype(bool)


def _scatter_to_map(shape, ys, xs, values):
    output = np.zeros(shape, dtype=np.float32)
    output[ys, xs] = values
    return output


def _to_uint8_mask(mask):
    return ((mask > 0).astype(np.uint8) * 255)


def _to_heatmap(values):
    clipped = np.clip(values, 0.0, 1.0)
    return cv2.applyColorMap((clipped * 255).astype(np.uint8), cv2.COLORMAP_JET)


def _polygon_canvas(shape, polygon):
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if polygon is not None and len(polygon) > 0:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    return canvas


def _ensure_rgb(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def _ensure_bgr(image):
    rgb = _ensure_rgb(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
