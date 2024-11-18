import json
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
"""
clip segmentation、筛掉bbox任一边<1多边形、重计算bbox、重命名ann_id
"""
for split in ['train','val','test1','test2']:
    ann_pth=f'whu_mix/{split}/annotation.json'
    out_pth=f'whu_mix/{split}/ann.json'
    coco=COCO(ann_pth)
    with open(ann_pth,'r') as f:
        data = json.load(f)
    valid_anns = []
    for ann in tqdm(data["annotations"]):
        segmentation = ann['segmentation'][0]            
        coco_img=coco.loadImgs(ann['image_id'])[0]
        poly = np.array(segmentation).reshape((-1, 2))
        # clip范围内：
        poly=np.clip(poly,0,coco_img['width']-1e-4)        
        xmin, ymin = np.min(poly, axis=0)
        xmax, ymax = np.max(poly, axis=0)
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        if bbox[2]>1 and bbox[3]>1:
            ann['bbox'] = [round(x,2) for x in bbox]
            ann['segmentation']=[poly.reshape(-1).tolist()]
            valid_anns.append(ann)
    n_all=len(data["annotations"])
    invalid=n_all-len(valid_anns)
    print(f'{split} total {n_all}, invalid {invalid}, {invalid/n_all*100:.2f}%')
    # 重新赋值annotation的id
    for i, ann in enumerate(valid_anns):
        ann['id'] = i   
    # 使用新的annotations列表替换原有的annotations
    data['annotations'] = valid_anns #验证：
    assert len(data['annotations'])==(data['annotations'][-1]['id']+1), 'ann_id not continuous'

    with open(out_pth, 'w') as f:
        json.dump(data, f)   
    print(split,'done')