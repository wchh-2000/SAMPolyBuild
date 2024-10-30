from pycocotools.coco import COCO
import numpy as np
import json
from tqdm import tqdm
import cv2
def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)
    return I,U

def get_mask(coco, img,res_type='rle',score_th=None):
    annotation_ids = coco.getAnnIds(imgIds=img['id'])
    annotations = coco.loadAnns(annotation_ids)

    mask = np.zeros((img['height'], img['width']), dtype=np.uint8)

    for ann in annotations:
        if score_th is not None:
            if 'score_cls' in ann:
                if ann['score_cls'] < score_th:
                    continue
            elif 'score' in ann:
                if ann['score'] < score_th:
                    continue
        if res_type=='rle':
            mask += coco.annToMask(ann)
        else:#poly
            poly = ann['segmentation'][0]
            pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=1)

    return mask != 0

def compute_IoU(coco_gt, coco_dt,res_type='rle'):
    # coco_gt: COCO object with ground truth annotations
    # coco_dt: COCO object with predicted annotations coco_gt.loadRes(results_mask)

    image_ids = coco_gt.getImgIds()#catIds=coco_gt.getCatIds()
    I,U=0,0
    for image_id in tqdm(image_ids):
        img = coco_dt.loadImgs(image_id)[0]
        mask_pred = get_mask(coco_dt, img,res_type)#,score_th=0.2)
        mask_gti = get_mask(coco_gt, img)

        i,u = calc_IoU(mask_pred, mask_gti)#一张图像的完整mask的iou
        I+=i
        U+=u
    IoU=round(I/U*100,4)
    print("IoU: ",IoU)
    return IoU

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('res', type=str,default='',help='path to the result file')
    parser.add_argument('gt', type=str,default='',help='path to the ground truth file')
    parser.add_argument('--save', type=str, default=None, help='path to save the result')
    args=parser.parse_args()

    coco_gt = COCO(args.gt)
    # Predictions annotations
    submission_file = json.loads(open(args.res).read())
    coco_dt = coco_gt.loadRes(submission_file)
    iou=compute_IoU(coco_gt, coco_dt)
    if args.save is not None:
        with open(args.save,'a+') as f:
            f.write(f'IoU {iou}\n')