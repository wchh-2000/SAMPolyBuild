from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
import os
join = os.path.join
import cv2
from pycocotools.coco import COCO
import albumentations as A
import random
from utils.data_utils import *
def get_bbox_point(bbox, hflip=False, vflip=False, jitter=True,add_center=True,input_size=224):
    # 几何变换：随机缩放，随机裁剪到crop_bbox，最终垂直/水平翻转，相应的bbox和point也要变换
    #bbox ndarray x, y, w, h
    # crop_bbox (x1, y1, x2, y2) 左上右下
    #transform: 
    if hflip:
        bbox[0]=input_size-bbox[0]-bbox[2] #注意bbox[0]还是左上点
    if vflip:
        bbox[1]=input_size-bbox[1]-bbox[3]
    if jitter:
        # bbox坐标和point坐标都要加上随机扰动，扰动范围为bbox的w和h的1/10 w,h扰动范围1/20
        jitter_amount_bbox = np.array([bbox[2] / 20, bbox[3] / 20, bbox[2] / 20, bbox[3] / 20]) #10% noise
        # jitter_amount_bbox = np.array([bbox[2] *0.15, bbox[3] *0.15, bbox[2] *0.15, bbox[3] *0.15])#30% noise
        bbox += np.random.uniform(-jitter_amount_bbox, jitter_amount_bbox)    
    #取新的bbox中心点:
    if add_center:
        point = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
        if jitter:
            jitter_amount_point = np.array([bbox[2] / 10, bbox[3] / 10])            
            point += np.random.uniform(-jitter_amount_point, jitter_amount_point)
    #x,y,w,h->x1,y1,x2,y2
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
    bbox=np.clip(bbox,0,input_size-1e-4)#防止bbox出界
    # bbox*=2
    if add_center:
        # point*=2
        point=point.reshape(1,2)
        return bbox, point
    else:
        return bbox
class PromptDataset(Dataset):
    def __init__(self, dataset_pth:dict, mode='train',load_gt=True,
                 select_ann_ids=None,gaussian=True,add_edge=False,crop_noise=False,
                 jitter=True,bbox=True,mix_bbox_point=False,input_size=224,iterative=False, **kwargs):
        # dataset_pth: {'data_root':'/data/datasets/whu_build/train','ann_file':'ann.json','img_dir':'images'}
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.input_size=input_size
        #监督信号参数：
        self.load_gt=load_gt#是否有gt
        self.add_edge=add_edge#是否同时用点和边监督
        self.gaussian=gaussian#是否生成高斯热力图
        #提示参数：
        self.crop_noise=crop_noise
        self.jitter=jitter#是否在训练时对提示关键点或bbox坐标进行随机偏移
        self.bbox=bbox#是否提示是bbox模式 bbox中心点与框（左上右下坐标点）一起作为提示 否则多点提示
        self.mix_bbox_point=mix_bbox_point
        #可能三种prompt_mode：bbox和中心点坐标；多点提示；bbox和多点提示混合
        self.iterative=iterative
        if self.bbox:
            self.prompt_mode='bbox'
        else:
            self.prompt_mode='point'
        self.mode = mode
        self.data_root=dataset_pth['data_root']
        self.coco_ann = COCO(join(self.data_root, dataset_pth['ann_file']))
        self.img_dir=join(self.data_root,dataset_pth['img_dir'])
        # Get image ids
        if select_ann_ids:
            self.ann_ids_f = select_ann_ids
        else:
            self.ann_ids_f = self.coco_ann.getAnnIds()
        self.ann_ids = []
        # ignore tiny objects:
        for ann_id in self.ann_ids_f:
            # ann = self.coco_ann.loadAnns(ann_id)[0]
            # if ann['bbox'][2]>=10 and ann['bbox'][3]>=10:# and len(ann['segmentation'][0])>6:
                #三角形len(ann['segmentation'][0])为8（含首尾）
            self.ann_ids.append(ann_id)
        # Define color transform
        self.color_transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            A.GaussNoise(p=0.5)
        ])

        print("Number of samples:", len(self.ann_ids),', ignore tiny objects:',len(self.ann_ids_f)-len(self.ann_ids))

    def __len__(self):
        return len(self.ann_ids)
    def __getitem__(self, index):
        ann_id=self.ann_ids[index]
        # Load annotation
        ann = self.coco_ann.loadAnns(ann_id)[0]
        img_id = ann['image_id']
        coco_img = self.coco_ann.loadImgs(img_id)[0]
        img_name = coco_img['file_name']
        self.ori_imgsize = (coco_img['height'], coco_img['width'])

        self.seg_h, self.seg_w = self.input_size,self.input_size #sam分割结果4倍上采样后计算损失
        self.gt_h, self.gt_w = int(self.input_size/4), int(self.input_size/4)#vmap,voff size. sam output feature map 1/4 
        # Read the image
        img = cv2.imread(join(self.img_dir, img_name))
        img = img[:, :, ::-1]  # BGR to RGB
        data = {}#返回的数据字典
        if self.mode == 'train':
            self.init_geometric_aug()
        bbox=ann['bbox']
        """
        crop_x, crop_y, crop_size = get_square_crop_bbox(bbox, self.ori_imgsize,rand=rand)#获取正方形裁剪框左上角坐标和边长
        # 裁剪图像
        cropped_img = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size,:]
        img = cv2.resize(cropped_img, (self.input_size, self.input_size))
        if self.mode == 'visualize':
            data['img_crop'] = img
        scale_factor = self.input_size / crop_size
        scale_factor_x=scale_factor_y=scale_factor
        bbox = np.array([(bbox[0] - crop_x) * (scale_factor), 
                (bbox[1] - crop_y) * (scale_factor), 
                bbox[2] * (scale_factor), 
                bbox[3] * (scale_factor)])
        # 保存坐标变换参数，用于恢复原始图像坐标
        pos_transform = [crop_x, crop_y, 1/scale_factor, 1/scale_factor]
        """
        crop_x, crop_y, crop_w,crop_h = get_crop_bbox(bbox, self.ori_imgsize,rand=self.crop_noise)#获取正方形裁剪框左上角坐标和边长
        # 裁剪图像
        cropped_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w,:]
        img = cv2.resize(cropped_img, (self.input_size, self.input_size))
        if self.mode == 'visualize':
            data['img_crop'] = img
        scale_factor_x = self.input_size / crop_w
        scale_factor_y = self.input_size / crop_h
        bbox = np.array([(bbox[0] - crop_x) * (scale_factor_x), 
                (bbox[1] - crop_y) * (scale_factor_y), 
                bbox[2] * (scale_factor_x), 
                bbox[3] * (scale_factor_y)])                
        # 保存坐标变换参数，用于恢复原始图像坐标
        pos_transform = [crop_x, crop_y, 1/scale_factor_x, 1/scale_factor_y]

        if self.load_gt:
            polygon = np.array(ann['segmentation'][0]).reshape(-1,2) #np.array(n,2)
            # 调整polygon和bbox的坐标到裁剪后的图像（input_size*input_size）上：
            polygon = (polygon - [crop_x, crop_y]) * [scale_factor_x,scale_factor_y]
            data['polygon']=polygon# input_size范围内，用于计算eval指标
            polygon=polygon*self.gt_h/self.input_size #resize range of polygon to gt_h, gt_w
            if self.mode == 'train':
                polygon=self.geometric_aug_polygons(polygon)
            data.update(self.get_batch_ann(polygon))

        if self.mode == 'train':
            img = self.geometric_aug_img(img)
            img = self.color_transform(image=img)['image']

        img = torch.as_tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1).contiguous()
        img = (img - self.pixel_mean) / self.pixel_std  
        data.update({'img': img, 'img_id': img_id})
        
        if self.mix_bbox_point:
            self.prompt_mode=random.choice(["bbox","mix"])#todo:不同batch point size不同情况处理
        # Get prompt points and their labels,prompt bbox:
        if self.mode == 'train':#只有训练时多种模式混合
            if self.prompt_mode=='bbox':
                bbox,points=get_bbox_point(bbox,self.hflip,self.vflip,self.jitter,input_size=self.input_size)
                label=np.array([1])
            else:#mix
                bbox,point_b=get_bbox_point(bbox,self.hflip,self.vflip,self.jitter,input_size=self.input_size)
                points,label=get_points_in_polygon(polygon,gt_size=self.gt_h,input_size=self.input_size)
                points=np.concatenate([point_b,points])
                label=np.concatenate([np.array([1]),label])
        else:#no transform
            if self.bbox:
                bbox,points=get_bbox_point(bbox,jitter=False,input_size=self.input_size)#
                label=np.array([1])
            else:
                points,label=get_points_in_polygon(polygon,gt_size=self.gt_h,input_size=self.input_size)
        
        data['points']=(torch.tensor(points).float(), torch.tensor(label).int())
        if self.prompt_mode=='bbox' or self.prompt_mode=="mix":            
            data['bbox']=torch.tensor(bbox).float()
        if self.mode!='train':
            data['pos_transform']=pos_transform
            data['ori_img_id']=img_id
            data['img_name']=img_name
        data['ori_size']=(self.input_size,self.input_size)
        data['ann_ids']=ann_id
        if self.iterative:
            data['iou_thr']=ann['iou_thr']
        return data
    def init_geometric_aug(self):
        self.hflip = random.choice([True, False]) #水平翻转
        self.vflip = random.choice([True, False]) #垂直翻转
    def geometric_aug_img(self, img):    
        if self.hflip:
            img = img[:, ::-1, :]#h,w,c
        if self.vflip:
            img = img[::-1, :, :]
        return img
    def geometric_aug_polygons(self,poly):
        #poly:  ndarray(n,2) (x,y) 对应 w,h gt_h,gt_w范围内
        if self.hflip:
            poly[:,0]=self.gt_w-poly[:,0]
        if self.vflip:
            poly[:,1]=self.gt_h-poly[:,1]                
        poly[:, 0] = np.clip(poly[:, 0], 0, self.gt_w - 1e-4)#减1e-4 防止对应到图像索引超界
        poly[:, 1] = np.clip(poly[:, 1], 0, self.gt_h - 1e-4)                
        return poly
    def get_batch_ann(self, polygon):  
        gt_mask=get_mask(polygon*(self.seg_h/self.gt_h), self.seg_h, self.seg_w)
        vmap, voff = get_point_ann(polygon,self.gt_h, self.gt_w,gaussian=self.gaussian)        
        if self.add_edge:
            edge=generate_edge_map(polygon, self.gt_h, self.gt_w,gaussian=self.gaussian)
        r={'gt_mask': torch.tensor(gt_mask).long().permute(2, 0, 1)}#[input_size, input_size, 1]->[1, input_size, input_size]
        if self.gaussian:
            vmap = torch.tensor(vmap).float()
        else:
            vmap = torch.tensor(vmap).long()
        r.update({
            'vmap': vmap.unsqueeze(0),
            'voff': torch.tensor(voff).float()
            })
        if self.add_edge:
            if self.gaussian:
                batch_edge= torch.tensor(edge).float()
            else:
                batch_edge = torch.tensor(edge).long()
            r['edge']=batch_edge 
        return r
def get_crop_bbox(bbox, img_size,rand=False):
    x0, y0, w0, h0 = bbox
    if not rand:
        # 扩展bbox区域15%： 
        side_expand=0.075
        expand=1+2*side_expand
        x=x0-w0*side_expand
        y=y0-h0*side_expand
        w=w0*expand
        h=h0*expand
        # x,y,w,h为裁剪区域左上角坐标和宽高
    else:
        # 一边扩展15%~30%，位置随机
        side_expand_l,side_expand_h=0.075,0.15
        expand_l,expand_h=1+2*side_expand_l,1+2*side_expand_h
        x=x0-w0*random.uniform(side_expand_l, side_expand_h)
        y=y0-h0*random.uniform(side_expand_l, side_expand_h)
        w=w0*random.uniform(expand_l, expand_h)
        h=h0*random.uniform(expand_l, expand_h)

    #处理边界：
    x=int(max(0,x))
    y=int(max(0,y))
    w=int(min(w,img_size[0]-x))
    h=int(min(h,img_size[1]-y))
    return x,y,w,h
def get_square_crop_bbox(bbox, img_size,rand=False):
    #rand:bbox在新框的位置是否随机
    w0, h0 = bbox[2], bbox[3]
    m = max(w0, h0)
    if m < 80:
        crop_size = int(m * 3)
    elif m < 120:
        crop_size = int(m * 1.8)
    else:
        crop_size = int(m * 1.2)
    if rand:
        ax = np.random.uniform(0.3, 0.7)
        ay = np.random.uniform(0.3, 0.7)
    else:
        ax,ay = 0.5,0.5
    # 确保裁剪框不超出图像边界，bbox在新框的位置随机，在（0.3-0.7 范围内）
    random_offset_x = ax * (crop_size - w0)
    random_offset_y = ay * (crop_size - h0)
    # 初步计算裁剪框的位置
    x0 = bbox[0] - random_offset_x
    y0 = bbox[1] - random_offset_y
    
    # 确保裁剪框不超出图像边界并调整位置
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x0 + crop_size > img_size[0]:
        x0 = img_size[0] - crop_size
    if y0 + crop_size > img_size[1]:
        y0 = img_size[1] - crop_size
    
    # 确保裁剪框包含原始bbox
    if x0 > bbox[0]:
        x0 = max(0, bbox[0] - (crop_size - w0))
    if y0 > bbox[1]:
        y0 = max(0, bbox[1] - (crop_size - h0))
    if x0 + crop_size < bbox[0] + w0:
        x0 = min(img_size[0] - crop_size, bbox[0] + w0 - crop_size)
    if y0 + crop_size < bbox[1] + h0:
        y0 = min(img_size[1] - crop_size, bbox[1] + h0 - crop_size)
    
    return int(x0), int(y0),crop_size
def collate_fn_test(batch):
    new_dict = {}
    for key in batch[0].keys():
        if key=='ori_size':
            new_dict[key] = batch[0][key] #所有图像尺寸一致input_size*input_size
        elif key=='points':
            points=[batch[i]['points'][0] for i in range(len(batch))]
            labels=[batch[i]['points'][1] for i in range(len(batch))]
            # 填充point
            max_point_num = max(len(point) for point in points)
            padded_points = []
            for point in points:
                padding = torch.zeros(max_point_num, 2)
                padding[:point.shape[0], :] = point
                padded_points.append(padding)

            # 填充label
            max_label_len = max(len(label) for label in labels)
            padded_labels = []
            for label in labels:
                padding = torch.full((max_label_len,), -1)
                padding[:len(label)] = label
                padded_labels.append(padding)

            # 返回填充后的point和label            
            new_dict[key] = (torch.stack(padded_points),torch.stack(padded_labels))
        elif key in ['img_id','img_name','ann_ids','polygon','pos_transform','scores','ori_img_id','iou_thr','img_crop']:
            new_dict[key] = [item[key] for item in batch]
        else:
            new_dict[key] = torch.stack([item[key] for item in batch], dim=0)
    return new_dict
