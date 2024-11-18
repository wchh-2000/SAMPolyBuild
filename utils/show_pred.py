import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pycocotools.mask import decode
from tqdm import tqdm
from os.path import join
def show_polygons(polygons,ax, color='blue', alpha=0.8):
    #alpha不透明度 0完全透明，1完全不透明
    for segment in polygons:
        # 绘制多边形轮廓
        segment = np.array(segment).reshape(-1, 2)
        ax.plot(segment[:, 0], segment[:, 1], color=color, alpha=alpha)
        # 使用scatter绘制多边形的顶点
        ax.scatter(segment[:, 0], segment[:, 1], color=color, s=2, alpha=alpha)
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 44/255, 30/255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_box(box, ax,score=None):
    x0, y0, w, h = box
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=1))
    if score is not None:
        # 在框的左上角添加得分
        score*=100
        text_location = (x0, y0)
        ax.text(*text_location, f"{score:.1f}", color='blue', backgroundcolor=(0, 0, 0, 0), fontsize=7)
def filt_boxes(boxes, scores, threshold):
    #boxes:xyxy
    #当一个得分较大的框与一个得分较小的框s的交集部分超过s面积*thr时，筛掉s
    
    # 计算每个框的面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 按分数从高到低排序
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算当前框与其他所有框的交集
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # 计算交集占其他框面积的比例
        overlap = inter / areas[order[1:]]
        
        # 移除重叠比例大于阈值的框
        inds = np.where(overlap <= threshold)[0]
        order = order[inds + 1]

    return boxes[keep], keep
def filter_boxes(dets, threshold=0.9,score_thr=0.1):
    #dets: list of anns in a img
    boxes = []
    scores = []
    for d in dets:
        # if d['score'] < score_thr:
        #     continue
        #xywh->xyxy
        box=d['bbox'].copy()
        box[2]+=box[0]
        box[3]+=box[1]
        boxes.append(box)
        scores.append(d['score'])
    if len(boxes)==0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    keep = filt_boxes(boxes, scores, threshold=threshold)[1]
    return [dets[i] for i in keep]
def show(img_dir,polygons,masks, image_ids, n_samples=None, show=False,img_suffix='.jpg'):
    """Visualization function."""
    score_thr = 0.3
    threshold=0.9
    bbox=True #是否含bbox检测结果（有bbox)
    if n_samples:
        # Uniform sampling
        ids = np.linspace(0, len(image_ids) - 1, n_samples, dtype=int)
        image_ids = [image_ids[i] for i in ids]
    
    for img_id in tqdm(image_ids):
        # Load the image
        img = plt.imread(join(img_dir,img_id+img_suffix))
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))        
        # Draw on left subplot (detection polygons overlay)
        ax1.imshow(img)
        ax1.axis('off')
        det_polygons=polygons[img_id]
        if bbox:
            det_polygons=filter_boxes(det_polygons,threshold,score_thr=score_thr)
        for det in det_polygons:
            if det['score'] < score_thr:  # filter low scores
                continue
            show_polygons(det['segmentation'], ax1, color='red')
        # Draw on right subplot (mask overlay)
        ax2.imshow(img)
        ax2.axis('off')
        det_masks=masks[img_id]
        if bbox:
            det_masks=filter_boxes(det_masks,threshold,score_thr=score_thr)
        for det in det_masks:
            if det['score'] < score_thr:  # filter low scores
                continue
            mask=decode(det['segmentation'])
            show_mask(mask, ax=ax2)
            if bbox:
                bbox=det['bbox']
                show_box(bbox, ax2, score=det['score'])
            
        # Remove margins
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if show:
            plt.show()
        else:
            plt.savefig(f'{vis_dir}{img_id}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()
def load_result(result_pth):
    with open(result_pth, 'r') as f:
        results = json.load(f)
    all_results = {}
    for res in results:
        img_id = res['image_id']
        if img_id not in all_results:
            all_results[img_id] = []
        all_results[img_id].append(res)
    return all_results
if __name__=='__main__':
    img_dir='dataset/whu_mix/test2/images/'
    dt_pth="work_dir/whumix_auto/results.json"
    img_suffix='.tif'
    dt_mask=dt_pth.replace('results','results_mask')
    vis_dir='/'.join(dt_pth.split('/')[:-1])+'/vis/' #保存在检测结果文件夹下
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
    polygons=load_result(dt_pth)
    masks=load_result(dt_mask)
    img_ids=list(polygons.keys())
    show(img_dir, polygons,masks, img_ids,n_samples=25,img_suffix=img_suffix)
