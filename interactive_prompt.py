# %% Required imports
from segment_anything import build_sam, SamPredictor
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

join = os.path.join
import argparse
import inspect
import json
from utils.test_utils import load_args
from utils.post_process import GetPolygons,transform_polygon_to_original
from utils.auto_result_bbox import infer_image_id, load_and_select_bbox
from utils.negative_feature_refine import refine_mask_by_negative_features, save_negative_feature_debug

# %% Helper functions
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='yellow', facecolor=(0,0,0,0), lw=1))

def show_points(coords, labels, ax, marker_size=100, label=''):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25, label=label)
    if neg_points.shape[0] > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def get_prompt(bbox, imgsize=512, prompt_point=None):
    x0, y0, xmax, ymax = bbox
    w0, h0 = xmax - x0, ymax - y0
    x, y = x0 - w0 * 0.075, y0 - h0 * 0.075
    w, h = w0 * 1.15, h0 * 1.15

    # Handle boundaries
    x = int(max(0, x))
    y = int(max(0, y))
    w = int(min(w, imgsize - x))
    h = int(min(h, imgsize - y))
    bbox_crop = [x, y, x + w, y + h]
    pos_transform = [x,y,1,1]#np.array([x,y,1,1],dtype=np.float32)

    new_bbox = [(x0 - x), (y0 - y), (w0), (h0)]
    point = np.array([new_bbox[0] + new_bbox[2] / 2, new_bbox[1] + new_bbox[3] / 2]).reshape(1, 2)

    if prompt_point is not None:
        prompt_point = (prompt_point - [x, y])
        point = np.concatenate([point, prompt_point], axis=0)

    new_bbox[2] = new_bbox[0] + new_bbox[2]
    new_bbox[3] = new_bbox[1] + new_bbox[3]

    return bbox_crop, point, np.array(new_bbox),pos_transform

# %% Set up the parser for model and task configuration
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='prompt_instance_spacenet')
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--imgpth', type=str, default='figs/eg.jpg')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--multi_mask', type=bool, default=True)
parser.add_argument('--max_distance', type=int, default=10)
parser.add_argument('--auto_results', type=str, default=None,
                    help='Path to auto-mode results.json. If set, first click selects an auto bbox.')
parser.add_argument('--auto_image_id', type=str, default=None,
                    help='Image id in auto results. Defaults to image file stem.')
parser.add_argument('--auto_min_score', type=float, default=0.0,
                    help='Minimum auto bbox score_cls/score for click selection.')
parser.add_argument('--auto_max_center_distance', type=float, default=None,
                    help='Reject nearest auto bbox when its center is farther than this distance.')
parser.add_argument('--debug_prompt_points', action='store_true',
                    help='Print prompt point coordinates and labels through the interactive prediction chain.')
parser.add_argument('--refine_by_points', action='store_true',
                    help='Remove connected mask components that contain negative prompt points before polygon extraction.')
parser.add_argument('--point_refine_mode', choices=['connected_component', 'distance'],
                    default='connected_component',
                    help='Point-guided mask refinement mode used when --refine_by_points is set.')
parser.add_argument('--negative_margin', type=float, default=0.0,
                    help='Distance margin for distance point refinement. Larger values remove more pixels near negative points.')
parser.add_argument('--negative_feature_refine', action='store_true',
                    help='Use negative-point feature prototypes to remove visually similar mask pixels before polygon extraction.')
parser.add_argument('--negative_similarity_thr', type=float, default=0.75,
                    help='Similarity threshold for negative feature refinement.')
parser.add_argument('--negative_feature_patch_size', type=int, default=21,
                    help='Patch size around each negative point for feature prototype extraction.')
parser.add_argument('--negative_protect_radius', type=int, default=20,
                    help='Radius around positive points that negative feature refinement cannot remove.')
parser.add_argument('--negative_spatial_sigma', type=float, default=80.0,
                    help='Spatial falloff sigma for negative feature refinement. Use <=0 to disable spatial weighting.')
parser.add_argument('--negative_same_region_refine', action='store_true',
                    help='Remove mask pixels with Lab colors similar to negative-point masked patches.')
parser.add_argument('--negative_patch_radius', type=int, default=10,
                    help='Patch radius around each negative point for same-region suppression.')
parser.add_argument('--negative_color_distance_thr', type=float, default=3.0,
                    help='Normalized Lab distance threshold for negative same-region suppression.')
parser.add_argument('--negative_same_region_protect_radius', type=int, default=20,
                    help='Radius around positive points protected from negative same-region suppression.')
parser.add_argument('--negative_region_max_radius', type=float, default=50.0,
                    help='Maximum distance from any negative point for same-region suppression. Use <=0 to disable.')
parser.add_argument('--negative_min_mask_patch_pixels', type=int, default=20,
                    help='Minimum connected mask pixels required around a valid negative point.')
parser.add_argument('--negative_max_removed_ratio', type=float, default=0.45,
                    help='Maximum mask area ratio that same-region suppression may remove before rollback.')

args = load_args(parser,path='configs/prompt_instance_spacenet.json')
args.result_pth = f'{args.work_dir}/{args.task_name}/'
args.checkpoint = 'prompt_interactive.pth'
os.makedirs(args.result_pth, exist_ok=True)
mask_result_dir = join(args.result_pth, "interactive_masks")
os.makedirs(mask_result_dir, exist_ok=True)
interactive_result_path = join(args.result_pth, "interactive_results.json")

# %% Load the model
device = 'cuda:' + str(args.gpu)
sam_model = build_sam(use_poly=True, load_pl=True, **vars(args)).to(device)
sam_model.eval()
predictor = SamPredictor(sam_model, polygon=True)
global image, bbox_coords, prompt_coords
# Load image
image = cv2.imread(args.imgpth)
if image is None:
    raise ValueError(f"Unable to read image: {args.imgpth}")
image = image[:, :, ::-1]  # BGR to RGB
bbox_coords = []  # To store bounding box coordinates
selected_bbox = None
selected_auto_info = None
current_instance_key = None
interaction_results = {}
instance_artists = {}
instance_logits = {}
instance_refined_masks = {}
prompt_coords = []  # To store prompt point coordinates
prompt_labels = [1]  # To store prompt point labels (1 or 0). First point is bbox center by default
defined_bbox = False
auto_bbox_enabled = args.auto_results is not None
auto_image_id = args.auto_image_id or infer_image_id(args.imgpth)


def reset_interaction():
    global bbox_coords, selected_bbox, selected_auto_info, current_instance_key, prompt_coords, prompt_labels, defined_bbox
    if current_instance_key is not None:
        instance_logits.pop(current_instance_key, None)
        instance_refined_masks.pop(current_instance_key, None)
    bbox_coords = []
    selected_bbox = None
    selected_auto_info = None
    current_instance_key = None
    prompt_coords = []
    prompt_labels = [1]
    defined_bbox = False


def assign_prompt_label(label):
    if not prompt_coords:
        print("No prompt point is waiting for a label.")
        return
    if len(prompt_labels) == len(prompt_coords) + 1:
        prompt_labels[-1] = label
    else:
        prompt_labels.append(label)
    print(f"Label {label} assigned to prompt point: {prompt_coords[-1]}")
    print("Press Enter to predict, or click another prompt point.")


def select_auto_bbox(click_point):
    bbox, info = load_and_select_bbox(
        args.auto_results,
        click_point,
        image_id=auto_image_id,
        min_score=args.auto_min_score,
        max_center_distance=args.auto_max_center_distance,
    )
    bbox = [int(round(v)) for v in bbox]
    print(
        "Auto bbox selected: "
        f"{bbox}, contains_click={info['contains_click']}, "
        f"score_cls={info['score_cls']:.4f}, score={info['score']:.4f}"
    )
    return bbox, info


def normalize_xyxy(bbox):
    x1, y1, x2, y2 = bbox
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def make_instance_key(bbox, auto_info=None):
    image_id = auto_image_id or infer_image_id(args.imgpth) or "image"
    if auto_info is not None and auto_info.get("index") is not None:
        return f"{image_id}_auto_{int(auto_info['index']):04d}"
    bbox_part = "_".join(str(int(round(v))) for v in bbox)
    return f"{image_id}_bbox_{bbox_part}"


def prompt_points_to_json(points, labels):
    if points is None:
        return []
    items = []
    for point, label in zip(points, labels[1:]):
        items.append({"x": int(point[0]), "y": int(point[1]), "label": int(label)})
    return items


def debug_prompt_state(stage, coords=None, labels=None, extra=None):
    if not args.debug_prompt_points:
        return
    print(f"[debug_prompt_points] {stage}")
    if coords is not None:
        print(f"  point_coords = {np.asarray(coords).tolist()}")
    if labels is not None:
        print(f"  point_labels = {np.asarray(labels).tolist()}")
    if extra is not None:
        print(f"  {extra}")


def predict_with_optional_debug(predictor, point_coords, point_labels, box, multimask_output, mask_input=None):
    kwargs = dict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        mask_input=mask_input,
        multimask_output=multimask_output,
    )
    if args.debug_prompt_points:
        signature = inspect.signature(predictor.predict)
        if "debug_prompt_points" in signature.parameters:
            kwargs["debug_prompt_points"] = True
        else:
            print(
                "[debug_prompt_points] predictor.predict does not accept debug_prompt_points; "
                "inner predictor logs are unavailable in this environment."
            )
    return predictor.predict(**kwargs)


def save_binary_mask(path, mask):
    mask_to_save = ((mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(path, mask_to_save)


def save_point_refine_regions(path, original_mask, refined_mask, point_coords, point_labels):
    before = (original_mask > 0).astype(np.uint8)
    after = (refined_mask > 0).astype(np.uint8)
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
    cv2.imwrite(path, vis)


def split_prompt_points(point_coords, point_labels, width, height):
    positive_points = []
    negative_points = []
    if point_coords is None or point_labels is None:
        return positive_points, negative_points
    for point, label in zip(point_coords, point_labels):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if not (0 <= x < width and 0 <= y < height):
            print(f"Warning: prompt point ({x}, {y}) is outside mask bounds ({width}, {height}); skipped.")
            continue
        if int(label) == 1:
            positive_points.append((x, y))
        elif int(label) == 0:
            negative_points.append((x, y))
    return positive_points, negative_points


def refine_mask_by_connected_components(mask, point_coords, point_labels):
    if point_coords is None or point_labels is None:
        return mask, False
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"Expected mask shape (1, H, W), got {mask.shape}")

    original_mask = mask.copy()
    binary_mask = (mask[0] > 0).astype(np.uint8)
    num_components, labels_map = cv2.connectedComponents(binary_mask, connectivity=8)
    all_components = set(range(1, num_components))
    height, width = binary_mask.shape
    positive_points, negative_points = split_prompt_points(point_coords, point_labels, width, height)
    positive_components = set()
    negative_components = set()

    for x, y in positive_points:
        component_id = int(labels_map[y, x])
        if component_id > 0:
            positive_components.add(component_id)
    for x, y in negative_points:
        component_id = int(labels_map[y, x])
        if component_id > 0:
            negative_components.add(component_id)

    if positive_components:
        keep_components = positive_components
    else:
        keep_components = all_components - negative_components
    keep_components = keep_components - negative_components

    if not keep_components:
        print("Warning: point refinement removed all mask components; using original mask.")
        return original_mask, False

    refined_mask_2d = np.isin(labels_map, list(keep_components)).astype(mask.dtype)
    if np.count_nonzero(refined_mask_2d) == 0:
        print("Warning: point refinement produced an empty mask; using original mask.")
        return original_mask, False

    refined_mask = refined_mask_2d.reshape(1, height, width)
    removed_components = sorted(negative_components & all_components)
    print(
        "Connected-component point refinement applied: "
        f"positive_components={sorted(positive_components)}, "
        f"negative_components={sorted(negative_components)}, "
        f"removed_components={removed_components}, "
        f"kept_components={sorted(keep_components)}"
    )
    return refined_mask, True


def refine_mask_by_distance(mask, point_coords, point_labels, negative_margin):
    if point_coords is None or point_labels is None:
        return mask, False
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"Expected mask shape (1, H, W), got {mask.shape}")

    original_mask = mask.copy()
    binary_mask = (mask[0] > 0).astype(np.uint8)
    height, width = binary_mask.shape
    positive_points, negative_points = split_prompt_points(point_coords, point_labels, width, height)
    if not positive_points or not negative_points:
        print("Warning: distance point refinement requires at least one positive and one negative point; using original mask.")
        return original_mask, False

    ys, xs = np.where(binary_mask > 0)
    if xs.size == 0:
        print("Warning: distance point refinement received an empty mask; using original mask.")
        return original_mask, False

    pixels = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    pos = np.asarray(positive_points, dtype=np.float32)
    neg = np.asarray(negative_points, dtype=np.float32)
    dist_to_pos = np.sqrt(((pixels[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    dist_to_neg = np.sqrt(((pixels[:, None, :] - neg[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    keep_pixels = dist_to_pos + float(negative_margin) < dist_to_neg

    refined_mask_2d = np.zeros_like(binary_mask, dtype=mask.dtype)
    refined_mask_2d[ys[keep_pixels], xs[keep_pixels]] = 1
    if np.count_nonzero(refined_mask_2d) == 0:
        print("Warning: distance point refinement removed all mask pixels; using original mask.")
        return original_mask, False

    removed_pixels = int(xs.size - np.count_nonzero(refined_mask_2d))
    print(
        "Distance point refinement applied: "
        f"positive_points={positive_points}, negative_points={negative_points}, "
        f"negative_margin={negative_margin}, removed_pixels={removed_pixels}, kept_pixels={int(np.count_nonzero(refined_mask_2d))}"
    )
    return refined_mask_2d.reshape(1, height, width), True


def refine_mask_by_points(mask, point_coords, point_labels):
    if args.point_refine_mode == 'connected_component':
        return refine_mask_by_connected_components(mask, point_coords, point_labels)
    if args.point_refine_mode == 'distance':
        return refine_mask_by_distance(mask, point_coords, point_labels, args.negative_margin)
    raise ValueError(f"Unsupported point_refine_mode: {args.point_refine_mode}")


def polygon_from_mask(mask, pred_vmap, pred_voff, crop_w, crop_h):
    polygons, scores, valid_mask = GetPolygons(
        mask,
        pred_vmap,
        pred_voff,
        ori_size=(crop_w, crop_h),
        max_distance=args.max_distance,
    )
    return polygons[0], scores[0], valid_mask[0]


def get_cached_logit(instance_key):
    if instance_key is None:
        return None
    return instance_logits.get(instance_key)


def cache_best_logit(instance_key, logit, best_idx):
    if instance_key is None:
        return
    if logit is None:
        print("[iter_refine] Warning: predictor returned no low-res logit; cache skipped.")
        return
    logit_array = np.asarray(logit)
    if logit_array.ndim != 3:
        print(f"[iter_refine] Warning: expected logit shape [C,H,W], got {logit_array.shape}; cache skipped.")
        return
    if best_idx < 0 or best_idx >= logit_array.shape[0]:
        print(
            f"[iter_refine] Warning: best_idx={best_idx} is outside returned logit "
            f"shape {logit_array.shape}; cache skipped."
        )
        return
    cached_logit = logit_array[best_idx][None, :, :]
    instance_logits[instance_key] = cached_logit
    print(f"[iter_refine] cached_logit_shape={cached_logit.shape}")


def get_cached_refined_mask(instance_key):
    if instance_key is None:
        return None
    cached_mask = instance_refined_masks.get(instance_key)
    return None if cached_mask is None else cached_mask.copy()


def cache_refined_mask(instance_key, mask):
    if instance_key is None or mask is None:
        return
    mask_array = np.asarray(mask)
    if mask_array.ndim != 3 or mask_array.shape[0] != 1:
        print(f"[negative_region] Warning: skip refined mask cache with shape {mask_array.shape}.")
        return
    instance_refined_masks[instance_key] = mask_array.copy()


def split_labeled_points(point_coords, point_labels):
    if point_coords is None or point_labels is None:
        return [], []
    positive_points = []
    negative_points = []
    for point, label_value in zip(point_coords, point_labels):
        item = (int(round(float(point[0]))), int(round(float(point[1]))))
        if int(label_value) == 1:
            positive_points.append(item)
        elif int(label_value) == 0:
            negative_points.append(item)
    return positive_points, negative_points


def _normalize_mask_shape(mask):
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        return mask_array[0], True
    if mask_array.ndim == 2:
        return mask_array, False
    raise ValueError(f"Expected mask shape (1, H, W) or (H, W), got {mask_array.shape}")


def _extract_negative_lab_prototypes(
    lab_image,
    search_region,
    negative_points,
    patch_radius,
    min_mask_patch_pixels=20,
):
    height, width = search_region.shape
    prototypes = []
    skipped_points = []
    radius = max(1, int(patch_radius))
    min_pixels = max(1, int(min_mask_patch_pixels))
    for point in negative_points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if not (0 <= x < width and 0 <= y < height):
            reason = "out_of_bounds"
            skipped_points.append({"point": (x, y), "reason": reason})
            print(f"[negative_region] skip negative point={(x, y)}, reason={reason}")
            continue
        if not search_region[y, x]:
            reason = "outside_current_mask"
            skipped_points.append({"point": (x, y), "reason": reason})
            print(f"[negative_region] skip negative point={(x, y)}, reason={reason}")
            continue

        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(width, x + radius + 1)
        y2 = min(height, y + radius + 1)
        patch_lab = lab_image[y1:y2, x1:x2]
        patch_mask = search_region[y1:y2, x1:x2]
        patch_mask_uint8 = patch_mask.astype(np.uint8)
        mask_pixels = int(np.count_nonzero(patch_mask_uint8))
        if mask_pixels < min_pixels:
            reason = f"insufficient_mask_patch_pixels:{mask_pixels}<{min_pixels}"
            skipped_points.append({"point": (x, y), "reason": reason})
            print(f"[negative_region] skip negative point={(x, y)}, reason={reason}")
            continue

        _, patch_labels = cv2.connectedComponents(patch_mask_uint8, connectivity=8)
        local_x, local_y = x - x1, y - y1
        component_id = int(patch_labels[local_y, local_x])
        component_mask = patch_labels == component_id
        component_pixels = int(np.count_nonzero(component_mask)) if component_id > 0 else 0
        if component_pixels < min_pixels:
            reason = f"sparse_local_component:{component_pixels}<{min_pixels}"
            skipped_points.append({"point": (x, y), "reason": reason})
            print(f"[negative_region] skip negative point={(x, y)}, reason={reason}")
            continue

        pixels = patch_lab[component_mask]
        source = "masked_local_component"
        pixels = pixels.astype(np.float32)
        mean = pixels.mean(axis=0)
        std = np.maximum(pixels.std(axis=0), 8.0)
        prototypes.append({
            "mean": mean,
            "std": std,
            "point": (x, y),
            "source": source,
            "mask_pixels": component_pixels,
        })
    return prototypes, skipped_points


def _positive_protect_mask(shape, positive_points, protect_radius):
    protect_mask = np.zeros(shape, dtype=np.uint8)
    radius = int(protect_radius)
    if radius <= 0:
        return protect_mask.astype(bool)
    height, width = shape
    for x, y in positive_points:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(protect_mask, (int(x), int(y)), radius, 1, thickness=-1)
    return protect_mask.astype(bool)


def _negative_radius_mask(shape, negative_points, max_radius):
    radius_mask = np.ones(shape, dtype=bool)
    if max_radius is None or float(max_radius) <= 0:
        return radius_mask, False
    radius_mask = np.zeros(shape, dtype=np.uint8)
    radius = int(round(float(max_radius)))
    height, width = shape
    for x, y in negative_points:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(radius_mask, (int(x), int(y)), radius, 1, thickness=-1)
    return radius_mask.astype(bool), True


def _negative_connected_similar_mask(negative_similar_mask, negative_points):
    similar_uint8 = negative_similar_mask.astype(np.uint8)
    num_components, labels_map = cv2.connectedComponents(similar_uint8, connectivity=8)
    height, width = negative_similar_mask.shape
    negative_component_ids = set()
    for x, y in negative_points:
        if 0 <= x < width and 0 <= y < height:
            component_id = int(labels_map[y, x])
            if component_id > 0:
                negative_component_ids.add(component_id)
    if not negative_component_ids:
        return np.zeros_like(negative_similar_mask, dtype=bool), max(0, num_components - 1), []
    return np.isin(labels_map, list(negative_component_ids)), max(0, num_components - 1), sorted(negative_component_ids)


def _keep_positive_components_or_largest(mask_2d, positive_points):
    binary_mask = (mask_2d > 0).astype(np.uint8)
    if np.count_nonzero(binary_mask) == 0:
        return binary_mask.astype(bool), 0, []

    num_components, labels_map, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    component_count = max(0, num_components - 1)
    height, width = binary_mask.shape
    positive_components = set()
    for x, y in positive_points:
        if 0 <= x < width and 0 <= y < height:
            component_id = int(labels_map[y, x])
            if component_id > 0:
                positive_components.add(component_id)

    if positive_components:
        keep_components = positive_components
    elif component_count > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_components = {int(np.argmax(areas)) + 1}
    else:
        keep_components = set()

    if not keep_components:
        return np.zeros_like(binary_mask, dtype=bool), component_count, []
    return np.isin(labels_map, list(keep_components)), component_count, sorted(keep_components)


def _restore_mask_shape(mask_2d, dtype, had_channel_dim):
    restored = mask_2d.astype(dtype)
    if had_channel_dim:
        return restored.reshape(1, restored.shape[0], restored.shape[1])
    return restored


def refine_mask_by_negative_same_region(
    image_crop,
    mask,
    point_coords,
    point_labels,
    patch_radius=10,
    distance_thr=3.0,
    protect_radius=20,
    max_radius=50,
    min_mask_patch_pixels=20,
    max_removed_ratio=0.45,
    previous_mask=None,
    debug=False,
):
    mask_array = np.asarray(mask)
    mask_2d, had_channel_dim = _normalize_mask_shape(mask_array)
    mask_dtype = mask_array.dtype
    search_region = mask_2d > 0
    use_previous_refined_mask = False
    if previous_mask is not None:
        try:
            previous_2d, _ = _normalize_mask_shape(previous_mask)
            if previous_2d.shape == search_region.shape and np.count_nonzero(previous_2d) > 0:
                search_region = previous_2d > 0
                use_previous_refined_mask = True
            else:
                print(
                    "[negative_region] Warning: previous refined mask is empty or has an incompatible "
                    f"shape {previous_2d.shape}; using current model mask."
                )
        except ValueError as exc:
            print(f"[negative_region] Warning: invalid previous refined mask; using current model mask: {exc}")
    original_mask = search_region.copy()
    mask_area_before = int(np.count_nonzero(original_mask))

    positive_points, negative_points = split_labeled_points(point_coords, point_labels)
    user_positive_points = []
    if point_coords is not None and point_labels is not None:
        for index, (point, label_value) in enumerate(zip(point_coords, point_labels)):
            if index > 0 and int(label_value) == 1:
                user_positive_points.append(
                    (int(round(float(point[0]))), int(round(float(point[1]))))
                )
    subject_positive_points = user_positive_points or positive_points
    if not negative_points:
        if debug:
            print("[negative_region] no negative points, skip")
        return mask
    if mask_area_before == 0:
        print("[negative_region] Warning: input mask is empty; using original mask.")
        return mask

    lab_image = cv2.cvtColor(image_crop, cv2.COLOR_RGB2LAB)
    prototypes, skipped_negative_points = _extract_negative_lab_prototypes(
        lab_image,
        search_region,
        negative_points,
        patch_radius,
        min_mask_patch_pixels=min_mask_patch_pixels,
    )
    valid_negative_points = [prototype["point"] for prototype in prototypes]
    if not prototypes:
        fallback_reason = "no_valid_negative_points"
        print("[negative_region] Warning: no valid negative prototypes; using previous refined mask.")
        if debug:
            _print_negative_region_debug(
                negative_points, prototypes, mask_area_before,
                np.zeros_like(search_region, dtype=bool),
                np.zeros_like(search_region, dtype=bool), original_mask, 0, [],
                fallback=True, valid_negative_points=valid_negative_points,
                skipped_negative_points=skipped_negative_points,
                fallback_reason=fallback_reason,
                use_previous_refined_mask=use_previous_refined_mask,
            )
        return _restore_mask_shape(original_mask, mask_dtype, had_channel_dim)

    ys, xs = np.where(search_region)
    lab_pixels = lab_image[ys, xs].astype(np.float32)
    negative_similar_pixels = np.zeros(xs.shape[0], dtype=bool)
    for prototype in prototypes:
        normalized = (lab_pixels - prototype["mean"].reshape(1, 3)) / prototype["std"].reshape(1, 3)
        distances = np.sqrt((normalized ** 2).sum(axis=1))
        negative_similar_pixels |= distances <= float(distance_thr)

    negative_similar_mask = np.zeros_like(search_region, dtype=bool)
    negative_similar_mask[ys[negative_similar_pixels], xs[negative_similar_pixels]] = True
    negative_component_mask, similar_component_count, neg_component_ids = _negative_connected_similar_mask(
        negative_similar_mask,
        valid_negative_points,
    )
    if not neg_component_ids:
        fallback_reason = "no_reliable_negative_component"
        print("[negative_region] Warning: no negative point falls inside a similar component; using previous refined mask.")
        if debug:
            _print_negative_region_debug(
                negative_points, prototypes, mask_area_before, negative_similar_mask,
                np.zeros_like(search_region, dtype=bool), original_mask, 0, [],
                fallback=True, similar_component_count=similar_component_count,
                neg_component_ids=neg_component_ids, radius_limited=False,
                radius_pixels=0, positive_protect_pixels=0,
                removed_pixels_before_component_filter=0,
                valid_negative_points=valid_negative_points,
                skipped_negative_points=skipped_negative_points,
                fallback_reason=fallback_reason,
                use_previous_refined_mask=use_previous_refined_mask,
            )
        return _restore_mask_shape(original_mask, mask_dtype, had_channel_dim)

    radius_mask, radius_limited = _negative_radius_mask(search_region.shape, valid_negative_points, max_radius)
    positive_protect = _positive_protect_mask(
        search_region.shape,
        subject_positive_points,
        protect_radius,
    )
    remove_before_component_filter = negative_similar_mask & search_region & (~positive_protect)
    remove_mask = negative_component_mask & search_region & radius_mask & (~positive_protect)

    refined_mask = search_region.copy()
    refined_mask[remove_mask] = False
    if np.count_nonzero(refined_mask) == 0:
        fallback_reason = "empty_after_suppression"
        print("[negative_region] Warning: same-region suppression removed all mask pixels; using previous refined mask.")
        if debug:
            _print_negative_region_debug(
                negative_points, prototypes, mask_area_before, negative_similar_mask,
                remove_mask, original_mask, 0, [], fallback=True,
                similar_component_count=similar_component_count,
                neg_component_ids=neg_component_ids,
                radius_limited=radius_limited,
                radius_pixels=int(np.count_nonzero(radius_mask & search_region)),
                positive_protect_pixels=int(np.count_nonzero(positive_protect)),
                removed_pixels_before_component_filter=int(np.count_nonzero(remove_before_component_filter)),
                valid_negative_points=valid_negative_points,
                skipped_negative_points=skipped_negative_points,
                fallback_reason=fallback_reason,
                use_previous_refined_mask=use_previous_refined_mask,
            )
        return _restore_mask_shape(original_mask, mask_dtype, had_channel_dim)

    refined_mask = cv2.morphologyEx(
        refined_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    refined_mask, component_count_after, keep_components = _keep_positive_components_or_largest(
        refined_mask,
        subject_positive_points,
    )
    if np.count_nonzero(refined_mask) == 0:
        fallback_reason = "empty_after_component_filter"
        print("[negative_region] Warning: component filtering removed all mask pixels; using previous refined mask.")
        if debug:
            _print_negative_region_debug(
                negative_points, prototypes, mask_area_before, negative_similar_mask,
                remove_mask, original_mask, component_count_after, keep_components, fallback=True,
                similar_component_count=similar_component_count,
                neg_component_ids=neg_component_ids,
                radius_limited=radius_limited,
                radius_pixels=int(np.count_nonzero(radius_mask & search_region)),
                positive_protect_pixels=int(np.count_nonzero(positive_protect)),
                removed_pixels_before_component_filter=int(np.count_nonzero(remove_before_component_filter)),
                valid_negative_points=valid_negative_points,
                skipped_negative_points=skipped_negative_points,
                fallback_reason=fallback_reason,
                use_previous_refined_mask=use_previous_refined_mask,
            )
        return _restore_mask_shape(original_mask, mask_dtype, had_channel_dim)

    height, width = refined_mask.shape
    lost_positive_points = [
        (x, y) for x, y in subject_positive_points
        if not (0 <= x < width and 0 <= y < height and refined_mask[y, x])
    ]
    mask_area_after = int(np.count_nonzero(refined_mask))
    removed_ratio = (mask_area_before - mask_area_after) / max(mask_area_before, 1)
    fallback_reason = None
    if lost_positive_points:
        fallback_reason = "positive_point_lost"
        print(
            "[negative_region][warning] positive point lost, fallback: "
            f"{lost_positive_points}"
        )
    elif removed_ratio > float(max_removed_ratio):
        fallback_reason = "removed_ratio_too_large"
        print(
            "[negative_region][warning] removed_ratio too large, fallback: "
            f"{removed_ratio:.4f}>{float(max_removed_ratio):.4f}"
        )

    if fallback_reason is not None:
        if debug:
            _print_negative_region_debug(
                negative_points, prototypes, mask_area_before, negative_similar_mask,
                remove_mask, refined_mask, component_count_after, keep_components, fallback=True,
                similar_component_count=similar_component_count,
                neg_component_ids=neg_component_ids,
                radius_limited=radius_limited,
                radius_pixels=int(np.count_nonzero(radius_mask & search_region)),
                positive_protect_pixels=int(np.count_nonzero(positive_protect)),
                removed_pixels_before_component_filter=int(np.count_nonzero(remove_before_component_filter)),
                valid_negative_points=valid_negative_points,
                skipped_negative_points=skipped_negative_points,
                removed_ratio=removed_ratio,
                fallback_reason=fallback_reason,
                use_previous_refined_mask=use_previous_refined_mask,
            )
        return _restore_mask_shape(original_mask, mask_dtype, had_channel_dim)

    if debug:
        _print_negative_region_debug(
            negative_points, prototypes, mask_area_before, negative_similar_mask,
            remove_mask, refined_mask, component_count_after, keep_components, fallback=False,
            similar_component_count=similar_component_count,
            neg_component_ids=neg_component_ids,
            radius_limited=radius_limited,
            radius_pixels=int(np.count_nonzero(radius_mask & search_region)),
            positive_protect_pixels=int(np.count_nonzero(positive_protect)),
            removed_pixels_before_component_filter=int(np.count_nonzero(remove_before_component_filter)),
            valid_negative_points=valid_negative_points,
            skipped_negative_points=skipped_negative_points,
            removed_ratio=removed_ratio,
            fallback_reason=None,
            use_previous_refined_mask=use_previous_refined_mask,
        )

    return _restore_mask_shape(refined_mask, mask_dtype, had_channel_dim)


def _print_negative_region_debug(
    negative_points,
    prototypes,
    mask_area_before,
    negative_similar_mask,
    remove_mask,
    refined_mask,
    component_count_after,
    keep_components,
    fallback,
    similar_component_count=0,
    neg_component_ids=None,
    radius_limited=False,
    radius_pixels=0,
    positive_protect_pixels=0,
    removed_pixels_before_component_filter=0,
    valid_negative_points=None,
    skipped_negative_points=None,
    removed_ratio=None,
    fallback_reason=None,
    use_previous_refined_mask=False,
):
    if neg_component_ids is None:
        neg_component_ids = []
    if valid_negative_points is None:
        valid_negative_points = []
    if skipped_negative_points is None:
        skipped_negative_points = []
    means = [prototype["mean"].round(2).tolist() for prototype in prototypes]
    stds = [prototype["std"].round(2).tolist() for prototype in prototypes]
    sources = [prototype["source"] for prototype in prototypes]
    print("[negative_region] enabled=True")
    print(f"[negative_region] negative_points={negative_points}")
    print(f"[negative_region] valid_negative_points={valid_negative_points}")
    print(f"[negative_region] skipped_negative_points={skipped_negative_points}")
    print(f"[negative_region] prototype_count={len(prototypes)}")
    print(f"[negative_region] prototype_lab_mean={means}")
    print(f"[negative_region] prototype_lab_std={stds}")
    print(f"[negative_region] prototype_sources={sources}")
    print("[negative_region] positive protection prefers user positive points; bbox center is fallback-only.")
    print(f"[negative_region] mask_area_before={mask_area_before}")
    print(f"[negative_region] negative_similar_pixels={int(np.count_nonzero(negative_similar_mask))}")
    print(f"[negative_region] similar_component_count={similar_component_count}")
    print(f"[negative_region] neg_component_ids={neg_component_ids}")
    print(f"[negative_region] radius_limited={radius_limited}")
    print(f"[negative_region] radius_pixels={radius_pixels}")
    print(f"[negative_region] positive_protect_pixels={positive_protect_pixels}")
    print(f"[negative_region] removed_pixels_before_component_filter={removed_pixels_before_component_filter}")
    print(f"[negative_region] removed_pixels_after_component_filter={int(np.count_nonzero(remove_mask))}")
    print(f"[negative_region] mask_area_after={int(np.count_nonzero(refined_mask))}")
    if removed_ratio is None:
        removed_ratio = (
            (mask_area_before - int(np.count_nonzero(refined_mask)))
            / max(mask_area_before, 1)
        )
    print(f"[negative_region] area_before={mask_area_before}")
    print(f"[negative_region] area_after={int(np.count_nonzero(refined_mask))}")
    print(f"[negative_region] removed_ratio={removed_ratio:.4f}")
    print(f"[negative_region] component_count_after={component_count_after}")
    print(f"[negative_region] keep_components={keep_components}")
    print(f"[negative_region] fallback={fallback}")
    print(f"[negative_region] fallback_reason={fallback_reason}")
    print(f"[negative_region] use_previous_refined_mask={use_previous_refined_mask}")


def save_interaction_result(instance_key, bbox, prompt_points, labels, polygon, mask):
    mask_name = f"{instance_key}_latest_mask.png"
    mask_path = join(mask_result_dir, mask_name)
    mask_to_save = ((mask[0] > 0).astype(np.uint8) * 255)
    cv2.imwrite(mask_path, mask_to_save)

    previous = interaction_results.get(instance_key, {})
    version = int(previous.get("version", 0)) + 1
    interaction_results[instance_key] = {
        "selected_bbox": [int(v) for v in bbox],
        "click_points": prompt_points_to_json(prompt_points, labels),
        "latest_polygon": [[float(x), float(y)] for x, y in polygon.tolist()],
        "latest_mask_path": f"interactive_masks/{mask_name}",
        "version": version,
    }
    payload = {
        "image_name": os.path.basename(args.imgpth),
        "instances": interaction_results,
    }
    with open(interactive_result_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Interactive result saved to {interactive_result_path} (instance={instance_key}, version={version})")


def clear_instance_artists(instance_key):
    for artist in instance_artists.get(instance_key, []):
        try:
            artist.remove()
        except ValueError:
            pass
    instance_artists[instance_key] = []


def draw_latest_prediction(instance_key, bbox, prompt_point, labels, polygon):
    clear_instance_artists(instance_key)
    artists = []
    if prompt_point is not None:
        pos_points = prompt_point[labels[1:] == 1]
        neg_points = prompt_point[labels[1:] == 0]
        if pos_points.shape[0] > 0:
            artists.append(
                ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                           s=300, edgecolor='white', linewidth=1.25)
            )
        if neg_points.shape[0] > 0:
            artists.append(
                ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
                           s=300, edgecolor='white', linewidth=1.25)
            )
    x0, y0 = bbox[0], bbox[1]
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    artists.append(
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='yellow', facecolor=(0, 0, 0, 0), lw=1))
    )
    line, = ax.plot(polygon[:, 0], polygon[:, 1], color='r', linewidth=1)
    artists.append(line)
    artists.append(ax.scatter(polygon[:, 0], polygon[:, 1], color='b', linewidths=1, marker='.'))
    instance_artists[instance_key] = artists
    fig.canvas.draw_idle()


def on_click(event):
    global bbox_coords, selected_bbox, selected_auto_info, current_instance_key, prompt_coords, prompt_labels, defined_bbox
    if event.xdata is not None and event.ydata is not None:
        click_point = (int(event.xdata), int(event.ydata))
        if auto_bbox_enabled and not defined_bbox:
            try:
                selected_bbox, selected_auto_info = select_auto_bbox(click_point)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Auto bbox selection failed: {exc}")
                return
            current_instance_key = make_instance_key(selected_bbox, selected_auto_info)
            prompt_coords.append(click_point)
            prompt_labels.append(1)
            defined_bbox = True
            print(f"Positive prompt point added: {click_point}")
            print("Click more prompt points and press 1/0 to label them, or press Enter to predict.")
        elif auto_bbox_enabled and defined_bbox:
            prompt_coords.append(click_point)
            print(f"Selected prompt point: {prompt_coords[-1]}")
            print("Now press 1 for positive or 0 for negative label for the clicked point.")
        elif len(bbox_coords) < 2:
            # Collect bounding box points
            bbox_coords.append(click_point)
            print(f"Selected bbox point: {bbox_coords[-1]}")
        else:
            # Collect prompt point
            prompt_coords.append(click_point)
            print(f"Selected prompt point: {prompt_coords[-1]}")
            print("Now press 1 for positive or 0 for negative label for the clicked point.")
        
        if len(bbox_coords) == 2 and not defined_bbox:
            print("Bounding box defined. Click on prompt point or press Enter to predict.")
            defined_bbox = True

def on_key(event):
    global image, bbox_coords, selected_bbox, selected_auto_info, current_instance_key, prompt_coords, prompt_labels, defined_bbox
    if event.key in ['0', '1'] :
        # Capture label for the latest prompt point
        assign_prompt_label(int(event.key))
    if event.key == 'c':
        reset_interaction()
        print("Interaction cleared.")
        return
    if event.key == 'enter' and (selected_bbox is not None or len(bbox_coords) == 2):
        # Make prediction
        print("Making prediction...")
        if selected_bbox is not None:
            bbox = selected_bbox
        else:
            bbox = normalize_xyxy([bbox_coords[0][0], bbox_coords[0][1], bbox_coords[1][0], bbox_coords[1][1]])
            if current_instance_key is None:
                current_instance_key = make_instance_key(bbox)
        prompt_point = np.array([prompt_coords[0]]) if prompt_coords else None
        label = np.array([1, 1]) if prompt_coords else np.array([1])
        if len(prompt_coords) > 0:
            if len(prompt_labels) != len(prompt_coords) + 1:
                print("Please press 1 or 0 to label the latest prompt point before predicting.")
                return
            prompt_point = np.array(prompt_coords)
            label = np.array(prompt_labels)
        else:
            prompt_point = None
            label = np.array([1])  # Default to positive if no prompt points
        raw_click_points = [
            (int(point[0]), int(point[1]), int(point_label))
            for point, point_label in zip(prompt_coords, label[1:])
        ]
        debug_prompt_state(
            "raw click points",
            coords=[point[:2] for point in raw_click_points],
            labels=[point[2] for point in raw_click_points],
            extra=f"raw_click_points = {raw_click_points}"
        )
        bbox_crop, point, new_bbox,pos_transform = get_prompt(bbox, imgsize=image.shape[1], prompt_point=prompt_point)
        debug_prompt_state(
            "after crop transform",
            coords=point,
            labels=label,
            extra=f"bbox_crop={bbox_crop}, new_bbox={new_bbox.tolist()}, pos_transform={pos_transform}"
        )
        image_crop = image[bbox_crop[1]:bbox_crop[3], bbox_crop[0]:bbox_crop[2], :]

        # Set image and predict
        predictor.set_image_resize(image_crop)
        debug_prompt_state("before predictor.predict", coords=point, labels=label)
        mask_input = get_cached_logit(current_instance_key)
        print(f"[iter_refine] instance={current_instance_key}, using_mask_input={mask_input is not None}")
        if mask_input is not None:
            print(f"[iter_refine] mask_input_shape={np.asarray(mask_input).shape}")
        mask, score, logit, pred_poly = predict_with_optional_debug(
            predictor=predictor,
            point_coords=point,
            point_labels=label,
            box=new_bbox,
            multimask_output=args.multi_mask,
            mask_input=mask_input,
        )
        print(f"[iter_refine] returned_logit_shape={np.asarray(logit).shape if logit is not None else None}")
        pred_vmap, pred_voff = pred_poly['vmap'], pred_poly['voff']
        pred_vmap = torch.sigmoid(pred_vmap)
        pred_voff = torch.sigmoid(pred_voff)
        crop_w, crop_h = bbox_crop[2] - bbox_crop[0], bbox_crop[3] - bbox_crop[1]
        best_idx = int(np.argmax(score)) if args.multi_mask else 0
        mask = mask[best_idx, :, :].reshape(1, crop_h, crop_w)
        cache_best_logit(current_instance_key, logit, best_idx)

        if args.refine_by_points:
            before_path = join(args.result_pth, "mask_before_point_refine.png")
            after_path = join(args.result_pth, "mask_after_point_refine.png")
            regions_path = join(args.result_pth, "mask_point_refine_regions.png")
            save_binary_mask(before_path, mask[0])
            original_mask_for_refine = mask.copy()
            mask, refined = refine_mask_by_points(mask, point, label)
            save_binary_mask(after_path, mask[0])
            save_point_refine_regions(regions_path, original_mask_for_refine[0], mask[0], point, label)
            if refined:
                print(f"Point refinement debug masks saved: {before_path}, {after_path}, {regions_path}")

        if args.negative_same_region_refine:
            previous_refined_mask = get_cached_refined_mask(current_instance_key)
            mask = refine_mask_by_negative_same_region(
                image_crop=image_crop,
                mask=mask,
                point_coords=point,
                point_labels=label,
                patch_radius=args.negative_patch_radius,
                distance_thr=args.negative_color_distance_thr,
                protect_radius=args.negative_same_region_protect_radius,
                max_radius=args.negative_region_max_radius,
                min_mask_patch_pixels=args.negative_min_mask_patch_pixels,
                max_removed_ratio=args.negative_max_removed_ratio,
                previous_mask=previous_refined_mask,
                debug=args.debug_prompt_points,
            )

        polygon_before_feature_refine = None
        if args.negative_feature_refine:
            before_feature_mask = mask[0].copy()
            polygon_before_feature_refine, _, _ = polygon_from_mask(mask, pred_vmap, pred_voff, crop_w, crop_h)
            positive_points, negative_points = split_labeled_points(point, label)
            feature_refine = refine_mask_by_negative_features(
                image_crop,
                before_feature_mask,
                positive_points,
                negative_points,
                patch_size=args.negative_feature_patch_size,
                similarity_thr=args.negative_similarity_thr,
                protect_radius=args.negative_protect_radius,
                spatial_sigma=args.negative_spatial_sigma,
            )
            mask = feature_refine["mask"].reshape(1, crop_h, crop_w)
            print(feature_refine["message"])

        # Post-process from the final mask, after all enabled refinements.
        cache_refined_mask(current_instance_key, mask)
        polygon_crop, score, _ = polygon_from_mask(mask, pred_vmap, pred_voff, crop_w, crop_h)
        if args.negative_feature_refine:
            save_negative_feature_debug(
                args.result_pth,
                before_feature_mask,
                mask[0],
                feature_refine["similarity_map"],
                feature_refine["prototypes"],
                polygon_before=polygon_before_feature_refine,
                polygon_after=polygon_crop,
            )
            print(f"Negative feature refinement debug images saved to {args.result_pth}")

        polygon = transform_polygon_to_original(polygon_crop, pos_transform)
        save_interaction_result(current_instance_key, bbox, prompt_point, label, polygon, mask)

        # Visualization
        draw_latest_prediction(current_instance_key, bbox, prompt_point, label, polygon)
        plt.axis('off')
        plt.savefig(join(args.result_pth, 'result.jpg'), bbox_inches='tight', pad_inches=0)
        plt.show()
        if auto_bbox_enabled:
            print("Prediction complete. Add positive/negative prompt points and press Enter again, or press c to clear.")
        else:
            reset_interaction()
            print("Prediction complete. Click on two points (top left and bottom right) to define bounding box.")
def onmousemove(event):
    # 处理鼠标移动事件，获取鼠标位置
    ix, iy = event.xdata, event.ydata

    # 如果鼠标在图像区域外，不更新
    if ix is None or iy is None:
        return

    # 更新水平和垂直线的位置
    hline.set_ydata([iy] * 2)
    vline.set_xdata([ix] * 2)

    ax.set_title(f'(x, y): {int(ix)}, {int(iy)}')

    # 重新绘制
    fig.canvas.draw_idle()
# %% Display image and connect events
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)

hline = Line2D([0, image.shape[1]], [0, 0], color='red', lw=1, linestyle='--')
vline = Line2D([0, 0], [0, image.shape[0]], color='red', lw=1, linestyle='--')
ax.add_line(hline)
ax.add_line(vline)
if auto_bbox_enabled:
    print(f"Auto bbox mode enabled. Using image_id={auto_image_id}. Click one building to select a bbox.")
else:
    print("Click on two points (top left and bottom right) to define bounding box.")
cid_move = fig.canvas.mpl_connect('motion_notify_event', onmousemove)
cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
#不显示白边框：
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
