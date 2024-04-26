"""
原始数据集含图像文件夹image和coco标注格式的ann.json。
用每个图像的实例bbox外框裁剪出图像小块。
bbox四周总共扩展0.15倍的w,h尺寸，整个小块reize到input_size*input_size。bbox在小块的位置随机
存储新的小块图像，创建新的cropped.json,计算裁剪后的标注segmentation和bbox坐标（在小块内的图像坐标下）
图像标注记录pos_transform四参数，左上坐标点和x,y两个轴的尺度因子，用于小块内的图像坐标转换为原始图像坐标
"""
import os
import json
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
def transform_coords_to_original(x, y, pos_transform):
    """
    Convert coordinates from the cropped image to the original image.
    pos_transform:[x,y,scale_x,scale_y] x,y为左上角坐标，scale_x,scale_y为x,y轴的尺度因子
    """
    orig_x = x * pos_transform[2] + pos_transform[0]
    orig_y = y * pos_transform[3] + pos_transform[1]
    return orig_x, orig_y

import random
dataset='spacenet'
input_size=224
# edge_thr=4#长/宽小于edge_thr的ann不要
save_crop=True
for split in ['train','val','test']:
    img_dir=f'dataset/{dataset}/{split}/images'
    cropped_img_dir=f'dataset/{dataset}/{split}/cropped_images'
    ann_pth=f'dataset/{dataset}/{split}/ann.json'
    out_ann_pth=f'dataset/{dataset}/{split}/cropped.json'
    os.makedirs(cropped_img_dir,exist_ok=True)
    with open(ann_pth, 'r') as f:
        annotations = json.load(f)
    coco = COCO(ann_pth)

    cropped_annotations = {
        "images": [],
        "annotations": [],
        "categories": annotations["categories"]
    }
    for image_info in tqdm(annotations['images']):
        if save_crop:
            image_path = os.path.join(img_dir, image_info['file_name'])
            image = cv2.imread(image_path)
        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            x0, y0, w0, h0 = ann['bbox']
            # if w0<edge_thr or h0<edge_thr:
            #     continue

            # 扩展bbox区域15%： 
            side_expand=0.075
            expand=1+2*side_expand
            x=x0-w0*side_expand
            y=y0-h0*side_expand
            w=w0*expand
            h=h0*expand
            # x,y,w,h为裁剪区域左上角坐标和宽高

            #一边扩展15%~30%
            # side_expand_l,side_expand_h=0.075,0.15
            # expand_l,expand_h=1+2*side_expand_l,1+2*side_expand_h
            # x=x0-w0*random.uniform(side_expand_l, side_expand_h)
            # y=y0-h0*random.uniform(side_expand_l, side_expand_h)
            # w=w0*random.uniform(expand_l, expand_h)
            # h=h0*random.uniform(expand_l, expand_h)

            #处理边界：
            x=int(max(0,x))
            y=int(max(0,y))
            w=int(min(w,image_info['width']-x))
            h=int(min(h,image_info['height']-y))
            new_img_name = f"{image_info['file_name'].split('.')[0]}_{ann['id']}.png"
            if save_crop:
                # 裁剪图像
                cropped = image[y:y+h, x:x+w, :]            
                # resize到input_size*input_size
                resized = cv2.resize(cropped, (input_size, input_size))            
                # 保存新的小块图像 {原始图像前缀}_{annid}
                save_path = os.path.join(cropped_img_dir, new_img_name)
                cv2.imwrite(save_path, resized)
            
            # 添加图像条目到cropped.json
            cropped_image_info = {
                "id": ann['id'],#裁剪图像id与标注id一致，一对一
                "file_name": new_img_name,
                "width": input_size,
                "height": input_size            
            }
            cropped_annotations["images"].append(cropped_image_info)

            # 修改标注的segmentation和bbox坐标
            scale_factor_x, scale_factor_y = w/input_size, h/input_size
            pos_transform=[x,y,scale_factor_x,scale_factor_y]#位置坐标变换参数
            #裁剪区域左上角坐标和x,y轴的尺度因子

            new_segmentation = []
            for segment in ann['segmentation']:
                new_segment = []
                for i in range(0, len(segment), 2):
                    new_segment.append((segment[i] - x) / scale_factor_x)
                    new_segment.append((segment[i+1] - y) / scale_factor_y)
                new_segmentation.append(new_segment)
            
            new_bbox = [
                (x0 - x) / scale_factor_x,
                (y0 - y) / scale_factor_y,
                w0 / scale_factor_x,
                h0 / scale_factor_y
            ]
            #检验恢复到原始bbox:
            cx,cy=transform_coords_to_original(new_bbox[0],new_bbox[1],pos_transform)
            assert abs(cx-x0)<1e-4 and abs(cy-y0)<1e-4, 'bbox not match'

            # 添加标注条目到cropped.json
            new_ann = {
                "id": ann['id'],
                "image_id": ann['id'],
                "ori_img_id":image_info['id'],#原始图像id
                "category_id": ann["category_id"],
                "segmentation": [[round(x,2) for x in new_segmentation[0]]],
                "bbox": [round(x,2) for x in new_bbox],
                "area": ann["area"],  # 按照原始面积
                "iscrowd": ann["iscrowd"],
                "pos_transform": pos_transform
            }
            cropped_annotations["annotations"].append(new_ann)

    with open(out_ann_pth, "w") as f:
        json.dump(cropped_annotations, f)
