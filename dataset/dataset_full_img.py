from torch.utils.data import Dataset,DataLoader
import numpy as np
from numpy.random import randint
import torch
import os
join = os.path.join
import cv2
from pycocotools.coco import COCO
import albumentations as A
import random
from utils.data_utils import *
class PromptDataset(Dataset):
    def __init__(self, dataset_pth:dict, mode='train',input_size=1024,
                 anns_per_sample=40,select_img_ids=None,
                 gaussian=False,add_edge=False,
                 jitter=True,bbox=True,mix_bbox_point=False):
        # dataset_pth: {'data_root':'/data/datasets/whu_build/train','ann_file':'ann.json','img_dir':'images'}
        #一个样本最多标注数anns_per_sample
        #图像上采样到input_size*input_size后输入SAM。训练时几何变换，先resize后统一到512,再上采样到input_size
        self.pixel_mean =  torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.anns_per_sample = anns_per_sample
        #监督信号参数：
        self.add_edge=add_edge#是否同时用点和边监督
        self.gaussian=gaussian#是否生成高斯热力图
        #提示参数：
        self.jitter=jitter#是否在训练时对提示关键点或bbox坐标进行随机偏移
        self.bbox=bbox#是否提示是bbox模式 bbox中心点与框（左上右下坐标点）一起作为提示 否则多点提示
        self.mix_bbox_point=mix_bbox_point
        #可能三种prompt_mode：bbox和中心点坐标；多点提示；bbox和多点提示 同时指定bbox=True时测试时固定bbox模式，否则多点模式
        if self.bbox:
            self.prompt_mode='bbox'
        else:
            self.prompt_mode='point'
        self.mode = mode
        self.input_size=input_size#输入图像尺寸
        self.gt_size=int(input_size/4)#vmap,voff尺寸
        self.data_root=dataset_pth['data_root']
        self.coco_ann = COCO(join(self.data_root, dataset_pth['ann_file']))
        self.img_dir=join(self.data_root,dataset_pth['img_dir'])

        # Get image ids
        if select_img_ids:
            self.img_ids = select_img_ids
        else:
            self.img_ids = self.coco_ann.getImgIds()

        # Create a sample list
        self.img_ann_list = []
        for img_id in self.img_ids:
            ann_ids = self.coco_ann.getAnnIds(imgIds=img_id)
            if ann_ids:  # ignore images without annotations
                # Distribute ann_ids into chunks
                full_chunks, remainder = divmod(len(ann_ids), self.anns_per_sample)
                for i in range(full_chunks):
                    self.img_ann_list.append([img_id, ann_ids[i*self.anns_per_sample:(i+1)*self.anns_per_sample]])
                if remainder:
                    self.img_ann_list.append([img_id, ann_ids[-remainder:]])

        # Define color transform
        self.color_transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            # A.GaussNoise(p=0.5) #todo delete
        ])

        print("Number of samples:", len(self.img_ann_list))

    def __len__(self):
        return len(self.img_ann_list)
    def __getitem__(self, index):
        img_id, ann_ids = self.img_ann_list[index]
        coco_img = self.coco_ann.loadImgs(img_id)[0]
        img_name = coco_img['file_name']
        self.ori_imgsize = (coco_img['height'], coco_img['width'])

        self.seg_h, self.seg_w = self.ori_imgsize
        self.gt_h, self.gt_w = self.gt_size,self.gt_size#vmap,voff size

        # Read the image
        img = cv2.imread(join(self.img_dir, img_name))
        img = img[:, :, ::-1]  # BGR to RGB
        data = {}#返回的数据字典
        if self.mode == 'train':
            self.init_geometric_aug()
        # Load annotations
        anns = self.coco_ann.loadAnns(ann_ids)
        polygons = [np.array(ann['segmentation'][0]).reshape(-1,2) for ann in anns]#list of np.array(n,2)
        if self.mode == 'train':
            polygons=self.geometric_aug_polygons(polygons)
            #找出非空的polygon:
            idxs=[i for i, x in enumerate(polygons) if x.any()]
            polygons=[p for p in polygons if p.any()]
            if not polygons:
                print('empty polygon')
                return None
            anns=[anns[i] for i in idxs]#anns与polygons一一对应，因为后面还要获取提示bbox或点
        else:# no transform, but resize range of polygon to gt_h, gt_w
            gt_scale_rate=self.gt_h/self.ori_imgsize[0]
            polygons=[p*gt_scale_rate for p in polygons]
        data=self.get_batch_ann(polygons)
        if self.mode == 'train':
            # Perform geometric and color transformation:
            img = self.geometric_aug_img(img)
            img = self.color_transform(image=img)['image']

        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        img = torch.as_tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1).contiguous()[None, :, :, :]
        img = (img - self.pixel_mean) / self.pixel_std  
        data.update({'img': img, 'img_id': img_id})
        
        if self.mix_bbox_point:
            self.prompt_mode=random.choice(["point","bbox","mix"])
        # Get prompt points and their labels,prompt bbox:
        batch_points, batch_label ,batch_bbox = [], [],[]
        for i,ann in enumerate(anns): #ann与polygons[i]一一对应
            if self.mode == 'train':#只有训练时多种模式混合
                if self.resize_shape[0] > 512:#crop
                    crop_bbox=self.crop_bbox
                else:
                    crop_bbox=None
                if self.prompt_mode=='bbox':
                    bbox,points=get_bbox_point(ann,self.scale_rate,crop_bbox,self.hflip,self.vflip,self.jitter,
                                               input_size=self.input_size)
                    label=np.array([1])
                elif self.prompt_mode=='point':
                    points, label = get_points_in_polygon(polygons[i],gt_size=self.gt_size,input_size=self.input_size)
                else:#mix
                    bbox,point_b=get_bbox_point(ann,self.scale_rate,crop_bbox,self.hflip,self.vflip,self.jitter,
                                               input_size=self.input_size)
                    points,label=get_points_in_polygon(polygons[i],gt_size=self.gt_size,input_size=self.input_size)
                    points=np.concatenate([points,point_b])
                    label=np.concatenate([label,np.array([1])])
            else:#no transform
                if self.bbox:
                    bbox,points=get_bbox_point(ann,scale_rate=512/self.ori_imgsize[0],jitter=False,
                                               input_size=self.input_size)
                    label=np.array([1])
                else:
                    points, label = get_points_in_polygon(polygons[i],gt_size=self.gt_h,input_size=self.input_size)
            if self.prompt_mode=='bbox' or self.prompt_mode=="mix":
                batch_bbox.append(bbox)                
            batch_points.append(points)
            batch_label.append(label)

        batch_points = np.concatenate(batch_points).reshape((len(anns), points.shape[0], 2))
        batch_label = np.concatenate(batch_label).reshape((len(anns), points.shape[0]))
        data['points']=(torch.tensor(batch_points).float(), torch.tensor(batch_label).int())
        if self.prompt_mode=='bbox' or self.prompt_mode=="mix":
            batch_bbox=np.concatenate(batch_bbox).reshape((len(anns),4))
            data['bbox']=torch.tensor(batch_bbox).float()
        data['ori_size']=self.ori_imgsize
        data['polygon']=polygons
        data['ori_img_id']=img_id
        return data
    def init_geometric_aug(self,scale=False):
        #先随机缩放，然后随机裁剪，最终垂直/水平翻转，同时作用在原图、polygon顶点、提示点坐标上
        if not scale:
            if self.ori_imgsize[0] != 512:
                self.scale_rate=512/self.ori_imgsize[0]
            else:
                self.scale_rate = 1#
            self.resize_shape=(512,512)
        else:
            if self.ori_imgsize[0]>512:#650
                self.scale_rate=np.random.uniform(0.8, 1.2)
            else:
                self.scale_rate=np.random.uniform(1.0, 1.6)
            self.resize_shape = (int(self.ori_imgsize[0] * self.scale_rate), int(self.ori_imgsize[1] * self.scale_rate))
        self.hflip = random.choice([True, False]) #水平翻转
        self.vflip = random.choice([True, False]) #垂直翻转
        
        # Random crop or just upscale #padding
        if self.resize_shape[0] < 512:
            self.resize_shape=512,512
            self.scale_rate=512/self.ori_imgsize[0]
        #crop bbox在geometric_aug_polygons时确定，保证bbox包含至少一个polygon
    def geometric_aug_img(self, img):
        img = cv2.resize(img, self.resize_shape, interpolation=cv2.INTER_LINEAR)
        if self.resize_shape[0] > 512:
            img = img[self.crop_h_start:self.crop_h_start + 512, self.crop_w_start:self.crop_w_start + 512, :]        
        if self.hflip:
            img = img[:, ::-1, :]#h,w,c
        if self.vflip:
            img = img[::-1, :, :]
        return img
    def geometric_aug_polygons(self,polys):
        #polys: list of ndarray(n,2) (x,y) 对应 w,h
        #resize:
        for poly in polys:
            poly*=self.scale_rate
        if self.resize_shape[0] > 512: # Randomly crop to 512*512 if the image is larger than 512
            flag=1
            while flag:
                self.crop_h_start = randint(0, self.resize_shape[0] - 512)
                self.crop_w_start = randint(0, self.resize_shape[1] - 512)
                self.crop_bbox=(self.crop_w_start, self.crop_h_start, self.crop_w_start+512, self.crop_h_start+512)
                flag+=1
                for poly in polys:
                    if clip_polygon_by_bbox(poly, self.crop_bbox).any():
                        flag=0#有一个poly在crop_bbox内就可以，结束循环，否则重新生成随机裁剪参数
                        break
                if flag>3:#已有2次不在crop_bbox内，指定bbox使其包含某poly的第一个点
                    x,y=polys[random.randint(0,len(polys)-1)][0]#np randint(0,0)会报错
                    self.crop_w_start,self.crop_h_start=point_in_box(int(x), int(y), self.resize_shape[1],self.resize_shape[0])
                    self.crop_bbox=(self.crop_w_start, self.crop_h_start, self.crop_w_start+512, self.crop_h_start+512)
                    flag=0
            for i in range(len(polys)):
                polys[i]=clip_polygon_by_bbox(polys[i], self.crop_bbox)
                if polys[i].any():
                    polys[i][:,0]=polys[i][:,0]-self.crop_w_start
                    polys[i][:,1]=polys[i][:,1]-self.crop_h_start
        for i in range(len(polys)):
            if polys[i].any():
                polys[i]=polys[i]/512*self.gt_h #边界范围512*512 -> gt_size*gt_size
                if self.hflip:
                    polys[i][:,0]=self.gt_w-polys[i][:,0]
                if self.vflip:
                    polys[i][:,1]=self.gt_h-polys[i][:,1]                
                polys[i][:, 0] = np.clip(polys[i][:, 0], 0, self.gt_w - 1e-4)#减1e-4 防止对应到图像索引超界
                polys[i][:, 1] = np.clip(polys[i][:, 1], 0, self.gt_h - 1e-4)
                
        return polys
    def get_batch_ann(self, polygons):
        # polygons: list of ndarray(n,2) (x,y) 对应 w,h gt_w,gt_h范围内
        batch_gt_mask, batch_vmap, batch_voff = [], [], []
        batch_edge=[]
        for polygon in polygons:
            # if self.mode == 'train':
            #     segm = transform_polygon(segm, self.ori_imgsize, self.scale_rate, self.hflip, self.vflip, self.crop_h_start, self.crop_w_start)
            # segm=[[s/2 for s in segm[0]]]            
            gt_mask=get_mask(polygon*(self.seg_h/self.gt_h), self.seg_h, self.seg_w)
            batch_gt_mask.append(gt_mask)
            vmap, voff = get_point_ann(polygon,self.gt_h, self.gt_w,gaussian=self.gaussian)
            batch_vmap.append(vmap)
            batch_voff.append(voff)
            if self.add_edge:
                edge=generate_edge_map(polygon, self.gt_h, self.gt_w,gaussian=self.gaussian)
                batch_edge.append(edge)
        batch_gt_mask = np.concatenate(batch_gt_mask).reshape((len(polygons), 1, self.seg_h, self.seg_w))
        batch_vmap = np.concatenate(batch_vmap).reshape((len(polygons), 1, self.gt_h, self.gt_w))
        batch_voff = np.concatenate(batch_voff).reshape((len(polygons), 2, self.gt_h, self.gt_w))
        if self.add_edge:
            batch_edge=np.concatenate(batch_edge).reshape((len(polygons), 1, self.gt_h, self.gt_w))
        r={'gt_mask': torch.tensor(batch_gt_mask).long()}
        if self.gaussian:
            vmap = torch.tensor(batch_vmap).float()
        else:
            vmap = torch.tensor(batch_vmap).long()
        r.update({
            'vmap': vmap,
            'voff': torch.tensor(batch_voff).float()
            })
        if self.add_edge:
            if self.gaussian:
                batch_edge= torch.tensor(batch_edge).float()
            else:
                batch_edge = torch.tensor(batch_edge).long()
            r['edge']=batch_edge
        return r

def collate_fn(batch):
    batch = [data for data in batch if data is not None]#去除None
    if not batch:
        return None
    new_dict = {}
    # 遍历batch中的第一个元素（一个字典）的所有键
    for key in batch[0].keys():
        # 使用torch.cat将同一个键对应的batch的值拼接在第0维度
        if key=='img_id':
            new_dict[key] = [item[key] for item in batch]
        elif key=='points':
            new_dict[key] = tuple([torch.cat([item[key][i] for item in batch], dim=0) for i in range(2)])
        else:
            new_dict[key] = torch.cat([item[key] for item in batch], dim=0)
    new_dict['instance_nums']=[len(item['points'][0]) for item in batch]#每个图像的实例数列表
    return new_dict
def collate_fn_test(batch):
    if not batch:
        return None
    return batch[0]
