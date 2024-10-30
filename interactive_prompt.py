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
from utils.test_utils import load_args
from utils.post_process import GetPolygons,transform_polygon_to_original

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

args = load_args(parser,path='configs/prompt_instance_spacenet.json')
args.result_pth = f'{args.work_dir}/{args.task_name}/'
args.checkpoint = 'prompt_interactive.pth'
os.makedirs(args.result_pth, exist_ok=True)

# %% Load the model
device = 'cuda:' + str(args.gpu)
sam_model = build_sam(use_poly=True, load_pl=True, **vars(args)).to(device)
sam_model.eval()
predictor = SamPredictor(sam_model, polygon=True)
global image, bbox_coords, prompt_coords
# Load image
image = cv2.imread(args.imgpth)
image = image[:, :, ::-1]  # BGR to RGB
bbox_coords = []  # To store bounding box coordinates
prompt_coords = []  # To store prompt point coordinates
prompt_labels = [1]  # To store prompt point labels (1 or 0). First point is bbox center by default
defined_bbox = False
def on_click(event):
    global bbox_coords, prompt_coords, prompt_labels,defined_bbox
    if event.xdata is not None and event.ydata is not None:
        if len(bbox_coords) < 2:
            # Collect bounding box points
            bbox_coords.append((int(event.xdata), int(event.ydata)))
            print(f"Selected bbox point: {bbox_coords[-1]}")
        else:
            # Collect prompt point
            prompt_coords.append((int(event.xdata), int(event.ydata)))
            print(f"Selected prompt point: {prompt_coords[-1]}")
            print("Now press 1 for positive or 0 for negative label for the clicked point.")
        
        if len(bbox_coords) == 2 and not defined_bbox:
            print("Bounding box defined. Click on prompt point or press Enter to predict.")
            defined_bbox = True

def on_key(event):
    global image, bbox_coords, prompt_coords, prompt_labels,defined_bbox
    if event.key in ['0', '1'] :
        # Capture label for the latest prompt point
        label = int(event.key)
        prompt_labels.append(label)
        print(f"Label {label} assigned to prompt point: {prompt_coords[-1]}")
        
        if len(prompt_coords) == len(prompt_labels):
            print("Prompt point and label assigned. Press Enter to predict.")
    if event.key == 'enter' and len(bbox_coords) == 2:
        # Make prediction
        print("Making prediction...")
        bbox = [bbox_coords[0][0], bbox_coords[0][1], bbox_coords[1][0], bbox_coords[1][1]]
        prompt_point = np.array([prompt_coords[0]]) if prompt_coords else None
        label = np.array([1, 1]) if prompt_coords else np.array([1])
        if len(prompt_coords) > 0:
            prompt_point = np.array(prompt_coords)
            label = np.array(prompt_labels)
        else:
            prompt_point = None
            label = np.array([1])  # Default to positive if no prompt points
        bbox_crop, point, new_bbox,pos_transform = get_prompt(bbox, imgsize=image.shape[1], prompt_point=prompt_point)
        image_crop = image[bbox_crop[1]:bbox_crop[3], bbox_crop[0]:bbox_crop[2], :]

        # Set image and predict
        predictor.set_image_resize(image_crop)
        mask, score, logit, pred_poly = predictor.predict(
            point_coords=point,
            point_labels=label,
            box=new_bbox,
            multimask_output=args.multi_mask,
        )
        pred_vmap, pred_voff = pred_poly['vmap'], pred_poly['voff']
        pred_vmap = torch.sigmoid(pred_vmap)
        pred_voff = torch.sigmoid(pred_voff)
        crop_w, crop_h = bbox_crop[2] - bbox_crop[0], bbox_crop[3] - bbox_crop[1]
        if args.multi_mask:
            mask = mask[np.argmax(score), :, :].reshape(1, crop_h, crop_w)

        # Post-process
        polygon, score, _ = GetPolygons(mask, pred_vmap, pred_voff, ori_size=(crop_w, crop_h),
                        max_distance=args.max_distance)
        polygon = polygon[0]
        polygon=transform_polygon_to_original(polygon, pos_transform)

        # Visualization
        if prompt_point is not None:
            show_points(prompt_point, label[1:], plt.gca(), marker_size=300)#label[0]为中心点
        show_box(bbox, plt.gca())
        plt.plot(polygon[:, 0], polygon[:, 1], color='r', linewidth=1)
        plt.scatter(polygon[:, 0], polygon[:, 1], color='b', linewidths=1, marker='.')
        plt.axis('off')
        plt.savefig(join(args.result_pth, 'result.jpg'), bbox_inches='tight', pad_inches=0)
        plt.show()
        bbox_coords = []
        prompt_coords = []
        prompt_labels = [1]
        defined_bbox = False
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
print("Click on two points (top left and bottom right) to define bounding box.")     
cid_move = fig.canvas.mpl_connect('motion_notify_event', onmousemove)
cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
#不显示白边框：
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
