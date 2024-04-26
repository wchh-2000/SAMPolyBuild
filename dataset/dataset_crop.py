import os
join = os.path.join
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
class PromptDataset(Dataset):
    def __init__(self, img_dir, result_file,input_size=224):
        #img_dir: Directory containing 1024x1024 images
        #result_file: Path to the COCO result file that contains bbox field
        self.img_dir = img_dir
        self.input_size=input_size
        self.crop_size=(self.input_size,self.input_size)#裁剪小图大小
        
        # Load bbox ann/detection file
        self.det=COCO(result_file)
        self.img_ids = self.det.getImgIds()
        #去除无bbox或bbox过小的图像：
        save_ids=[]
        for img_id in self.img_ids:
            c=0
            anns= self.det.loadAnns(self.det.getAnnIds(imgIds=img_id))
            for ann in anns:
                if ann['bbox'][2]>=2 and ann['bbox'][3]>=2:
                    c+=1
            if c>0:
                save_ids.append(img_id)
        self.img_ids=save_ids
        
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image_info = self.det.loadImgs(img_id)[0]
        img_name = image_info['file_name']
        # Read the image
        img = cv2.imread(join(self.img_dir, img_name))
        img = img[:, :, ::-1]  # BGR to RGB
        
        results_ids=self.det.getAnnIds(imgIds=img_id)
        results = self.det.loadAnns(results_ids)
        bboxes = [result['bbox'] for result in results]
        crops = []
        new_bboxes = []
        points=[]
        pos_transforms=[]
        ann_ids=[]
        for i,bbox in enumerate(bboxes):#一批数据包含一个图像中的所有实例
            x0, y0, w0, h0 = bbox
            side_expand=0.075
            x=x0-w0*side_expand
            y=y0-h0*side_expand
            w=w0*(1+2*side_expand)
            h=h0*(1+2*side_expand)
            #处理边界：
            x=int(max(0,x))
            y=int(max(0,y))
            w=int(min(w,image_info['width']-x))
            h=int(min(h,image_info['height']-y))
            if w<2 or h<2:
                continue
            # Crop the image
            cropped = img[y:y+h, x:x+w, :]
            resized = cv2.resize(cropped, (self.input_size, self.input_size))
            resized = torch.as_tensor(resized.copy(), dtype=torch.float32).permute(2, 0, 1).contiguous()
            resized = (resized - self.pixel_mean) / self.pixel_std
            crops.append(resized)

            #裁剪区域左上角坐标和x,y轴的尺度因子:
            scale_factor_x, scale_factor_y = w/self.input_size, h/self.input_size
            pos_transform=[x,y,scale_factor_x,scale_factor_y]#位置坐标变换参数
            new_bbox = [
                (x0 - x) / scale_factor_x,
                (y0 - y) / scale_factor_y,
                w0 / scale_factor_x,
                h0 / scale_factor_y
            ]
            point = np.array([new_bbox[0] + new_bbox[2] / 2, new_bbox[1] + new_bbox[3] / 2])
            #x,y,w,h->x1,y1,x2,y2
            new_bbox[2] = new_bbox[0] + new_bbox[2]
            new_bbox[3] = new_bbox[1] + new_bbox[3]
            ann_ids.append(results[i]['id'])
            new_bboxes.append(new_bbox)
            points.append(point)
            pos_transforms.append(pos_transform)
        
        # Concatenate all cropped images
        batch_crops = torch.stack(crops, dim=0)  # Concatenate along batch dimension
        new_bboxes=torch.tensor(np.array(new_bboxes))
        points=torch.tensor(np.array(points)).reshape(-1,1,2)
        labels=torch.ones(len(new_bboxes)).reshape(-1,1)
        if 'score' in results[0]:
            scores=[res['score'] for res in results]
        else:
            scores=torch.ones(len(new_bboxes)).reshape(-1,1)
        return dict(img=batch_crops, ori_img_id=[img_id]*len(new_bboxes), #ori_img_id和dataset_crop中长度一致
            bbox=new_bboxes,points=(points,labels),pos_transform=pos_transforms,scores=scores,ori_size=self.crop_size,ann_ids=ann_ids)
def collate_fn_test(batch):    
    if not batch:
        return None
    return batch[0]
