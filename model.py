from torch.nn import functional as F
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
join = os.path.join
#self defined:
from segment_anything import build_sam
from utils.post_process import GetPolygons,generate_coco_ann,transform_polygon_to_original
from utils.metrics.iou import IoU
from utils.test_utils import PolygonMetrics
from utils.losses import sigmoid_l1_loss,focal_dice_loss,BCEDiceLoss
class PromptModel(pl.LightningModule):
    def __init__(self, args,test_cfg=None,divide_by_area=False):
        super().__init__()
        self.args = args
        self.test_cfg=test_cfg
        self.results_poly = []
        self.no_mask_n=0
        if test_cfg.train:
            load_pl=False
        else:
            load_pl=True            
        self.sam_model = build_sam(load_pl=load_pl,**vars(args))
        self.vmap_loss = BCEDiceLoss()
        self.bound_loss = BCEDiceLoss(pos_weight=2)
        self.divide_by_area = divide_by_area
        self.metrics_calculator = PolygonMetrics(divide_by_area=divide_by_area)
        
    def forward_step(self, batch,seg_size):
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=batch.get('points',None),
                boxes=batch.get('bbox',None),
                masks=None,
            )
        image_embedding = self.sam_model.image_encoder(batch['img'])

        seg_prob, iou_predictions,pred_poly= self.sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.args.multi_mask,
        )
        seg_prob=F.interpolate(seg_prob, size=seg_size,
            mode='bilinear', align_corners=False)
        res=dict(seg=seg_prob,poly=pred_poly)
        if self.args.multi_mask:
            res['iou']=iou_predictions
        return res
    def training_step(self, batch, batch_idx):
        gt_mask=batch['gt_mask']
        res=self.forward_step(batch,seg_size=(gt_mask.shape[2],gt_mask.shape[3]))
        pred_poly,seg_logit=res['poly'],res['seg']
        if self.args.multi_mask:
            pred_ious=res['iou']
            iou_matrix = IoU(seg_logit, gt_mask)#[b,3,h,w],[b,1,h,w]->[b,3]
            loss_iou=F.mse_loss(pred_ious,iou_matrix)
            iou_max, indices = torch.max(iou_matrix, dim=1)  # iou_max: [b], indices: [b]
            batch_indices = torch.arange(seg_logit.size(0)).unsqueeze(1).to(indices.device)
            seg_logit = seg_logit[batch_indices, indices.unsqueeze(1)]

        loss_seg = focal_dice_loss(seg_logit, gt_mask)#只用最大iou的mask计算loss                    
        gt_vmap=batch['vmap']
        gt_voff=batch['voff']
        pred_vmap=pred_poly['vmap']
        pred_voff=pred_poly['voff']
        loss_vmap = self.vmap_loss(pred_vmap, gt_vmap)
        loss_voff = sigmoid_l1_loss(pred_voff,gt_voff,mask=gt_vmap)
        self.log('train/vmap_loss', loss_vmap, on_step=True, logger=True,prog_bar=True)
        self.log('train/voff_loss', loss_voff, on_step=True, logger=True)
        loss=loss_seg+loss_vmap+loss_voff
        if self.args.multi_mask:
            self.log('train/iou_loss', loss_iou, on_step=True, logger=True)
            loss=loss+loss_iou
        if self.args.add_edge:
            gt_edge=batch['edge']
            pred_edge=pred_poly['edge']
            loss_edge=0.5*self.bound_loss(pred_edge, gt_edge)
            loss=loss+loss_edge
            self.log('train/edge_loss', loss_edge, on_step=True, logger=True)
        self.log('train_loss', loss, on_step=True, logger=True)
        self.log('train/seg_loss', loss_seg, on_step=True, logger=True,prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx,log=True):
        ori_size=batch['ori_size']
        with torch.no_grad():
            res=self.forward_step(batch,seg_size=ori_size)
        pred_poly,seg_logit=res['poly'],res['seg']
        if self.test_cfg.eval:
            gt_mask=batch['gt_mask']
            iou_matrix = IoU(seg_logit, gt_mask)#[b,3/1,h,w],[b,1,h,w]->[b,3/1]
            if self.args.multi_mask:
                pred_ious=res['iou']
                loss_iou=F.mse_loss(pred_ious,iou_matrix)
                if log:
                    self.log('val/iou_loss', loss_iou, on_step=False, on_epoch=True, logger=True)
                # 找到每个样本的最大 IoU
                iou_matrix, _ = torch.max(iou_matrix, dim=1)
            miou_mask=sum(iou_matrix)/len(iou_matrix)
            if log:
                self.log('val/mIoUmask', miou_mask, on_step=False, on_epoch=True, logger=True,prog_bar=True)
            gt_polygons=batch['polygon']#实例框范围（input_size)内的多边形
        if self.args.multi_mask:
            pred_ious=res['iou']#[b,3]
            _ , indices = torch.max(pred_ious, dim=1)  # indices: [b]
            # 选择每个b的最大 pred IoU对应的 seg_logit
            #  0 到 b-1 的范围作为第一个维度的索引
            batch_indices = torch.arange(seg_logit.size(0)).unsqueeze(1).to(indices.device)
            # indices 作为第二个维度的索引
            seg_logit = seg_logit[batch_indices, indices.unsqueeze(1)]
        seg_prob = torch.sigmoid(seg_logit).cpu().numpy().squeeze(1)
        pred_vmap=pred_poly['vmap'].sigmoid()
        pred_voff=pred_poly['voff'].sigmoid()
        if self.args.instance_input:
            pos_transforms=batch['pos_transform']
            ori_img_ids=batch['ori_img_id']
        else:
            pos_transforms=None
            ori_img_id=batch['ori_img_id']#only one img per batch
        batch_polygons, batch_scores,valid_idx,no_mask=GetPolygons(
                seg_prob,pred_vmap,pred_voff,ori_size=ori_size,
                max_distance=self.args.max_distance,pos_transforms=pos_transforms)
        batch_n=len(seg_prob)
        if no_mask>0:#delete no mask pred instances
            assert no_mask==batch_n-len(valid_idx)
            self.no_mask_n+=no_mask            
            if self.test_cfg.eval:
                gt_polygons=[gt_polygons[i] for i in valid_idx]
            if self.args.instance_input:
                pos_transforms=[pos_transforms[i] for i in valid_idx]
                ori_img_ids=[ori_img_ids[i] for i in valid_idx]
        gt_size=pred_vmap.shape[-1]
        for b in range(len(batch_polygons)):
            if self.args.instance_input:
                pred_polygon=transform_polygon_to_original(batch_polygons[b], pos_transforms[b],allready_scale=True)#恢复到原图（650*650）坐标
            else:
                pred_polygon=batch_polygons[b]
            
            if self.test_cfg.eval:
                if self.args.instance_input:
                    gt_polygon=transform_polygon_to_original(gt_polygons[b], pos_transforms[b])
                else:
                    gt_polygon=gt_polygons[b]*ori_size[0]/gt_size
                self.metrics_calculator.calculate_metrics(pred_polygon, gt_polygon)
            if self.test_cfg.save_results:
                if self.args.instance_input:
                    ori_img_id=ori_img_ids[b]
                self.results_poly.append(generate_coco_ann(pred_polygon,batch_scores[b],ori_img_id))
        if log:
            m=self.metrics_calculator.compute_average(batch_n)#todo 多卡验证是否有问题
            if self.divide_by_area:
                for i,size in enumerate(['large', 'medium', 'small']):
                    for key in m[i]:
                        self.log(f'val_area/{key}_{size}', m[i][key], on_step=False, on_epoch=True, logger=True)
            else:                
                self.log('val/v-precision', m['precision'], on_step=False, on_epoch=True, logger=True)
                self.log('val/v-recall', m['recall'], on_step=False, on_epoch=True, logger=True)
                self.log('val/v-f1', m['vf1'], on_step=False, on_epoch=True, logger=True,prog_bar=True)
                self.log('val/mIoU', m['miou'], on_step=False, on_epoch=True, logger=True,prog_bar=True)
                self.log('val/bound_f', m['bound_f'], on_step=False, on_epoch=True, logger=True,prog_bar=True)
    
    def configure_optimizers(self):
        paramlrs=[]
        if not self.args.freeze_mask:
            paramlrs.append(
            {'params': self.sam_model.mask_decoder.parameters(), 'lr': self.args.decoder_lr})
        if not self.args.freeze_img:
            paramlrs.append({'params': self.sam_model.image_encoder.parameters(), 'lr': self.args.img_encoder_lr})
        optimizer = torch.optim.AdamW(paramlrs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0)
        scheduler = [{'scheduler': cosine_scheduler, 'interval': 'epoch', 'frequency': 1, 'name': 'cosine', 'strict': True}]
        return [optimizer], scheduler