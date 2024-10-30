import copy
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models import TwoStageDetector, StandardRoIHead, SinePositionalEncoding, FCNMaskHead
from mmdet.models.task_modules import SamplingResult,AssignResult
from mmdet.models.utils import unpack_gt_instances, empty_instances
from mmdet.structures import SampleList, DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask.mask_target import mask_target_single
from mmdet.structures.mask import BitmapMasks
from mmengine.config import Config
from mmdet.utils import InstanceList
from mmpl.registry import MODELS, TASK_UTILS
from mmengine.model import BaseModel, BaseModule
from einops import rearrange, repeat
from mmpl.utils import ConfigType, OptConfigType
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import multiprocessing as mp
import atexit
import time
#custom:
from utils.losses import sigmoid_l1_loss,BCEDiceLoss
from utils.post_process import GetPolygons
from utils.data_utils import get_point_ann,generate_edge_map

@MODELS.register_module()
class SAMAnchorInstanceHead(TwoStageDetector):
    def __init__(
            self,
            sam_head=True,
            neck: OptConfigType = None,
            rpn_head: OptConfigType = None,
            roi_head: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            weight=None,
            **kwargs
    ):
        super(TwoStageDetector, self).__init__()
        self.neck = MODELS.build(neck)
        self.sam_head = sam_head

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)#添加test_cfg
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.weight=weight

    def extract_feat(self, x):
        x = self.neck(x)
        return x

    def loss(self,
             batch_inputs,
             batch_data_samples: SampleList,
             sam
             ) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        img_seg_feat = batch_inputs[0]
        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                        self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)
        
        if self.sam_head:
            roi_losses = self.roi_head.loss(x, rpn_results_list,
                                            batch_data_samples,
                                            sam, img_seg_feat
                                            )
        else:
            roi_losses = self.roi_head.loss(x, rpn_results_list,
                                            batch_data_samples
                                            )
        losses.update(roi_losses)

        losses['loss_cls']*=self.weight['cls']
        losses['loss_bbox']*=self.weight['bbox']

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                sam,
                rescale: bool = True
                ) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)
        img_seg_feat = batch_inputs[0]

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(#mmdet.RPNHead
                x, batch_data_samples, rescale=False)
        #rpn_results_list[0].bboxes.shape=[max_per_img,4] 原图尺寸 scores.shape=[max_per_img]
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        if self.sam_head:#True
            results_list = self.roi_head.predict(#SAMAnchorPromptRoIHead
                x, rpn_results_list, batch_data_samples, sam, img_seg_feat, rescale=rescale)
        else:
            results_list = self.roi_head.predict(
                x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples#batch_data_samples[i].pred_instances.polygons 为第i个图像中的所有建筑物的多边形
        # res=results_list[0]
        # return res.bboxes,res.masks,res.labels,res.scores#for flops

@MODELS.register_module()
class SAMAnchorPromptRoIHead(StandardRoIHead):
    def __init__(
            self,
            positional_encoding=dict(num_feats=128, normalize=True),
            expand_roi: bool = False,            
            expand_scale: float = 0.1,#四周扩展expand_scale/2倍
            add_pe: bool = True,#是否在bbox_head输入特征加入位置编码
            max_distance=11,
            multi_process=False,
            *args,
            **kwargs
    ):
        super(StandardRoIHead, self).__init__(*args, **kwargs)
        self.generator_pe = SinePositionalEncoding(**positional_encoding)
        self.expand_roi = expand_roi
        self.expand_scale = expand_scale
        if 'roi_mask_sampler' in self.train_cfg:
            self.roi_mask_sampler= TASK_UTILS.build(
                    self.train_cfg.roi_mask_sampler, default_args=dict(context=self))
        self.add_pe=add_pe
        self.mask_size=self.train_cfg.mask_size #train_cfg.rcnn
        self.max_distance=max_distance
        self.multi_process=multi_process
        if self.multi_process:
            num_processes = 6  # Set the number of processes
            self.pool = mp.Pool(processes=num_processes)        
            atexit.register(self.pool.close)
        self.pos_process_time=0

    def _mask_forward(self,
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None,
                      sam=None, img_seg_feat=None#[图像batch数,256,64,64]
                      ) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.(b,256,128,128),(b,256,64,64),(b,256,32,32)
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:#rois: [n, 5] n个预测的实例（一个batch的图像的实例拼在一起）

            mask_feats = self.mask_roi_extractor(
                [img_seg_feat], rois)#mask_feats: [n, 256, 14, 14] 
            #ALTER:[img_seg_feat] mask_roi_extractor: featmap_strides=[16]
            #ORIGIN:x[:self.mask_roi_extractor.num_inputs]
            if self.with_shared_head:#False
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats, sam)#mask_head: SAMPromptMaskHead mask_feats为sam decoder的输入roi特征
        mask_results = dict(mask_preds=mask_preds[0], mask_iou=mask_preds[1], mask_feats=mask_feats,
                             poly_preds=mask_preds[2])
        return mask_results

    def mask_loss(self, 
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList,
                  sam, img_seg_feat
                  ) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        # if not self.share_roi_extractor:#默认share_roi_extractor=False
        if hasattr(self, 'roi_mask_sampler') and self.expand_roi:#有roi_mask_sampler时，sampling_results和bbox独立
            for res in sampling_results:
                res.pos_priors = expand_bboxes(res.pos_priors,scale=self.expand_scale)
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(#根据roi裁剪图像特征，然后送入mask_head得到推理结果
            pos_rois, sam=sam, img_seg_feat=img_seg_feat)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            poly_preds=mask_results['poly_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_results.update(loss=mask_loss_and_target)
        return mask_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample],
             sam, img_seg_feat
             ) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        x = list(x)
        if self.add_pe:
            bs, _, h, w = x[-1].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.generator_pe(mask_pe)
            for i in range(len(x)):
                x[i] = x[i] + torch.nn.functional.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear')

        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        losses = dict()
        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        sampling_mask_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            if hasattr(self, 'roi_mask_sampler'):
                assign_result_copy = AssignResult(#bbox_sampler中add_gt_as_proposals会增加assign_result,所以先复制一份
                    num_gts=assign_result.num_gts,
                    gt_inds=assign_result.gt_inds.clone(),
                    max_overlaps=assign_result.max_overlaps.clone(),
                    labels=assign_result.labels.clone() if assign_result.labels is not None else None
                )
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)#rpn产生的roi 得到的sample
            if hasattr(self, 'roi_mask_sampler'):
                sampling_mask_result = self.roi_mask_sampler.sample(
                    assign_result_copy,
                    rpn_results,
                    batch_gt_instances[i])
                sampling_mask_results.append(sampling_mask_result)#rpn产生的roi 得到的sample
        if not hasattr(self, 'roi_mask_sampler'):
            sampling_mask_results=sampling_results
        # bbox head loss
        if self.with_bbox:
            if self.expand_roi:
                for res in sampling_results:
                    res.pos_priors = expand_bboxes(res.pos_priors,scale=self.expand_scale)
                    res.neg_priors = expand_bboxes(res.neg_priors,scale=self.expand_scale)
                    # res.priors = expand_bboxes(res.priors,scale=self.expand_scale)
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])
        
        if self.with_mask:
            mask_results = self.mask_loss(sampling_mask_results,
                                #用sampling_results[i].pos_priors（assign sample后的）作为mask的roi
                                          batch_gt_instances,
                                          sam, img_seg_feat
                                          )
            losses.update(mask_results['loss'])

        return losses


    def predict_mask(self,
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False,
                     sam=None, img_seg_feat=None
                     ) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        for res,img_meta in zip(results_list,batch_img_metas):#res是引用，results_list内的会修改
            res.rois = expand_bboxes(res.bboxes,scale=self.expand_scale)#expand roi区域，用于mask_head预测
            scale_factor = res.bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))
            res.bboxes/= scale_factor #恢复bboxes到原始图像尺度
        # don't need to consider aug_test.
        bboxes = [res.rois for res in results_list]
        if sum([b.shape[0] for b in bboxes])==0:#for empty roi in all batch(cal flops)
            results_list = empty_instances(
                batch_img_metas,
                bboxes[0].device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list
        mask_rois = bbox2roi(bboxes)#list -> tensor[n,5] batch_ind, x1, y1, x2, y2
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(mask_rois, sam=sam, img_seg_feat=img_seg_feat)#已经resize到标签大小
        mask_preds = mask_results['mask_preds']
        poly_preds=mask_results['poly_preds']

        vmap_preds, voff_preds = poly_preds['vmap'], poly_preds['voff']
        masks = torch.sigmoid(mask_preds).squeeze(1).detach().cpu().numpy()
        vmaps = torch.sigmoid(vmap_preds)
        voffs = torch.sigmoid(voff_preds)

        pos_transforms = []
        ori_size=self.mask_size#roi_size*4 GetPolygons中使得vmaps和masks尺寸一致
        s=time.perf_counter()
        #get pos_transform for each roi:
        for img_meta, results in zip(batch_img_metas, results_list):
            if len(results) == 0:
                print(f"empty roi in {img_meta['img_path']}")
                continue
            rois=results.rois.cpu().numpy()#xyxy 是bboxes expand后的
            rois[:,[0,2]]/=img_meta['scale_factor'][0]#scale_factor: 1024/ori_shape 转换到原图尺寸
            rois[:,[1,3]]/=img_meta['scale_factor'][1]

            pos_transforms.extend([
                [bbox[0],bbox[1],(bbox[2]-bbox[0])/self.mask_size,(bbox[3]-bbox[1])/self.mask_size]
                  for bbox in rois])
            
        pool=self.pool if self.multi_process else None
        polygons, scores, valid_mask =GetPolygons(masks,vmaps,voffs,\
                        ori_size=(ori_size,ori_size),max_distance=self.max_distance,\
                        pos_transforms=pos_transforms,pool=pool)
        # 处理完成后，根据每张图像的ROI数量将结果分配到各图像的results中
        current_index = 0
        mask_results=[]
        for idx, (img_meta, results) in enumerate(zip(batch_img_metas, results_list)):
            roi_count = len(results)
            if roi_count == 0:
                mask_results.append(mask_preds[current_index:current_index])#空tensor
                continue
            # 当前图像提取结果：
            polygons_per_img = polygons[current_index:current_index + roi_count]
            scores_per_img = scores[current_index:current_index + roi_count]
            valid_per_img = valid_mask[current_index:current_index + roi_count]
            mask_per_img = mask_preds[current_index:current_index + roi_count]
            current_index += roi_count

            if not np.any(valid_per_img):
                print("empty polygon")
                results_list[idx] = empty_instances(
                    [img_meta],
                    vmaps.device,
                    task_type='mask',
                    mask_thr_binary=self.test_cfg.mask_thr_binary)[0]
            elif not np.all(valid_per_img):#有无效的结果
                results_list[idx]=results[valid_per_img]
                results_list[idx].polygons = [p for p in polygons_per_img if p is not None]
                results_list[idx].mask_scores = scores_per_img[valid_per_img]
                mask_results.append(mask_per_img[valid_per_img])
            else:
                results.polygons = polygons_per_img
                results.mask_scores = scores_per_img
                mask_results.append(mask_per_img)
        self.pos_process_time+=(time.perf_counter()-s)
        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(#SAMPromptMaskHead 调用 _predict_by_feat_single mask恢复到原始图像位置
            mask_preds=mask_results,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)#rescale=True
        return results_list

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                sam, img_seg_feat,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        x = list(x)
        if self.add_pe:
            bs, _, h, w = x[-1].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.generator_pe(mask_pe)
            for i in range(len(x)):
                x[i] = x[i] + torch.nn.functional.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear')

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        assert self.with_bbox, 'Bbox head must be implemented.'        

        # TODO: nms_op in mmcv need be enhanced, the bbox result may get
        #  difference when not rescale in bbox_head

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        if self.expand_roi:
            for res in rpn_results_list:#res是引用，rpn_results_list内的会修改
                res.bboxes = expand_bboxes(res.bboxes,scale=self.expand_scale)
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        if self.with_mask:
            results_list = self.predict_mask(#用roi bbox head预测结果裁剪特征预测mask
                batch_img_metas, results_list, rescale=rescale, sam=sam, img_seg_feat=img_seg_feat)

        return results_list
def expand_bboxes(bboxes, w=1024, h=1024, scale=0.1):#todo 根据bboxes获取方形roi,不改变长宽比
    """
    bboxes: tensor [n, 4] where each row is a bbox in xyxy format
    w, h: int, image dimensions, bboxes should not exceed these dimensions
    scale: float, the expansion factor, bboxes are expanded by scale*0.5 on each side
    """
    # Calculate bbox widths and heights
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    
    # Calculate the amount to expand bboxes
    expand_widths = widths * scale / 2
    expand_heights = heights * scale / 2
    
    # Expand bboxes
    expanded_bboxes = torch.stack([
        bboxes[:, 0] - expand_widths,
        bboxes[:, 1] - expand_heights,
        bboxes[:, 2] + expand_widths,
        bboxes[:, 3] + expand_heights
    ], dim=1)
    
    # Clip the bboxes to be within the image dimensions
    expanded_bboxes[:, 0] = torch.clamp(expanded_bboxes[:, 0], min=0)
    expanded_bboxes[:, 1] = torch.clamp(expanded_bboxes[:, 1], min=0)
    expanded_bboxes[:, 2] = torch.clamp(expanded_bboxes[:, 2], max=w)
    expanded_bboxes[:, 3] = torch.clamp(expanded_bboxes[:, 3], max=h)
    
    return expanded_bboxes
@MODELS.register_module()
class SAMPromptMaskHead(FCNMaskHead):

    def __init__(self,
                 class_agnostic=False,
                 expand_roi: bool = False,                 
                 expand_scale: float = 0.1,#四周扩展expand_scale/2倍
                 center_prompt: bool = True,          
                 add_edge=True,
                 loss_weight=dict(mask=0.5,edge=0.5,vmap=1,voff=1),
                 *args,
                 **kwargs
                 ) -> None:
        super(BaseModule, self).__init__()
        self.class_agnostic = class_agnostic
        self.mask_loss=BCEDiceLoss(pos_weight=2)
        self.vmap_loss=BCEDiceLoss()
        self.expand_roi=expand_roi
        self.expand_scale=expand_scale
        self.center_prompt=center_prompt
        self.add_edge=add_edge
        self.weight=loss_weight
    def get_prompt_embed(self,sam,device,n,roi_size=14):
        """
        n: roi个数
        roi_size: roi feature map尺寸
        """
        input_size=roi_size*16 #默认roi_size=14 时224*224尺寸 14（roi mask_size)*16（原图是特征图16倍）
        center=int(input_size/2)
        if self.center_prompt:
            center=torch.tensor([center,center],device=device).reshape(1,1,2)
            label=torch.tensor([1],device=device).reshape(1,1)
            point=(center,label)
        else:
            point=None
        if self.expand_roi:
            left_top=self.expand_scale/2/(1+self.expand_scale)
            right_bottom=1-left_top
            bbox_coords=[left_top,left_top,right_bottom,right_bottom]#todo 暂未考虑expand roi后超过图像边界，裁剪的roi的情况
        else:
            bbox_coords=[0,0,1,1]
        bbox=input_size*torch.tensor([bbox_coords],device=device)
        sam.prompt_encoder.image_embedding_size=(roi_size,roi_size)
        sam.prompt_encoder.input_image_size=(input_size,input_size)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=point,
            boxes=bbox,
            masks=None,
        )
        # n=n.to(device)
        #repeat sparse_embeddings n:
        sparse_embeddings=torch.repeat_interleave(sparse_embeddings,n,dim=0)
        dense_embeddings=torch.repeat_interleave(dense_embeddings,n,dim=0)        
        #sparse_embeddings [n, 3, 256]
        #dense_embeddings [n,256,14,14]
        return sparse_embeddings,dense_embeddings

    def forward(self, x, sam) -> Tensor:
        num_instance = x.shape[0]
        #使用中心点和外框提示编码： 
        point_emb, nomask_dense_embeddings = self.get_prompt_embed(sam,x.device,num_instance,roi_size=x.shape[-1])
        
        img_pe = sam.prompt_encoder.pe_layer(x.shape[-2:]).unsqueeze(0) #位置编码为roi feature map的位置编码，不是相对原图位置的 [1, 256, roi_size, roi_size]
        img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=num_instance)

        low_res_masks, iou_predictions,pred_poly = sam.mask_decoder(
            image_embeddings=x,
            image_pe=img_pe,
            sparse_prompt_embeddings=point_emb,
            dense_prompt_embeddings=nomask_dense_embeddings,
            multimask_output=False
        )
        mask_pred = low_res_masks
        iou_predictions = iou_predictions.squeeze(1)
        return mask_pred, iou_predictions,pred_poly

    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            Tensor: Mask target of each positive proposals in the image.
        """
        pos_proposals = [res.pos_priors for res in sampling_results]#xyxy形式预测的bbox list batch*[n,4] n不定 为该图像正样本数 
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results#预测bbox(正样本)对应的gt_instance的索引
        ]#list batch*[n]
        gt_imgs_polygons = [res.masks for res in batch_gt_instances]#list batch*BitmapMasks/PolygonMasks num_masks为该图像实例数
        mask_targets_list = []
        vmap_targets_list,voff_targets_list=[],[]
        edge_targets_list=[]
        mask_size = (rcnn_train_cfg.mask_size,) * 2 #区域mask标签的尺寸
        
        roi_cfg=Config({'mask_size': mask_size, 'soft_mask_target': True})
        h,w=256,256
        device = pos_proposals[0].device
        for rois, pos_gt_inds, gt_polygons in zip(pos_proposals, pos_assigned_gt_inds, gt_imgs_polygons):#遍历一个batch(各图像)
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
                print("empty pos_gt_inds")
            else:#gt_polygons为一个图像中的所有实例的轮廓多边形
                mask_targets = mask_target_single(
                    rois, pos_gt_inds, gt_polygons.to_bitmap(), roi_cfg
                )                
                vmaps,voffs,edges=[],[],[]
                for polygon in gt_polygons:
                    if len(polygon)!=1:
                        print("gt_masks not 1")
                    polygon=(polygon[0]/4).reshape(-1,2)#坐标范围1024->256
                    xmin,ymin=polygon.min(axis=0)
                    xmax,ymax=polygon.max(axis=0)
                    size=max(xmax-xmin,ymax-ymin)
                    if size>mask_size[0]:
                        sigma=size/mask_size[0]
                    else:
                        sigma=1
                    vmap,voff=get_point_ann(polygon,h,w,sigma=sigma)
                    vmaps.append(vmap)
                    voffs.append(voff)
                    if self.add_edge:
                        edges.append(generate_edge_map(polygon,h,w))
                rois_s=rois/4
                vmaps_np=np.array(vmaps)
                vmaps=BitmapMasks(vmaps_np,h,w)
                vmap_targets=mask_target_single(rois_s, pos_gt_inds, vmaps, roi_cfg)
                voff_x=BitmapMasks(np.array([voff[0,:,:] for voff in voffs]),h,w)
                voff_y=BitmapMasks(np.array([voff[1,:,:] for voff in voffs]),h,w)
                voff_x_t=mask_target_single(rois_s, pos_gt_inds, voff_x, roi_cfg)
                voff_y_t=mask_target_single(rois_s, pos_gt_inds, voff_y, roi_cfg)
                voff_targets=torch.stack([voff_x_t,voff_y_t],dim=1)
                # show_roi(rois_s.cpu(),vmaps_np,pos_gt_inds.cpu(),vmap_targets.cpu(),voff_targets.cpu())
                if self.add_edge:
                    edges=BitmapMasks(np.array(edges),h,w)
                    edge_targets=mask_target_single(rois_s, pos_gt_inds, edges, roi_cfg)
            mask_targets_list.append(mask_targets)
            vmap_targets_list.append(vmap_targets)
            voff_targets_list.append(voff_targets)
            if self.add_edge:
                edge_targets_list.append(edge_targets)
        mask_targets = torch.cat(mask_targets_list)
        vmap_targets = torch.cat(vmap_targets_list)
        voff_targets = torch.cat(voff_targets_list)
        target=dict(mask=mask_targets,vmap=vmap_targets,voff=voff_targets)
        if self.add_edge:
            edge_targets=torch.cat(edge_targets_list)
            target['edge']=edge_targets
        return target
        
    def loss_and_target(self, mask_preds: Tensor,
                        poly_preds: (Tensor,Tensor,Tensor),
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (num_pos, num_classes, h, w).
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss and targets components.
        """
        vmap_preds,voff_preds= poly_preds['vmap'],poly_preds['voff']
        
        target = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)
        mask_targets,vmap_targets,voff_targets= target['mask'],target['vmap'],target['voff']
        if self.add_edge:
            edge_targets=target['edge']
            edge_preds =poly_preds['edge']

        mask_preds = torch.nn.functional.interpolate(
            mask_preds, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False)
        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            mask_targets=mask_targets.unsqueeze(1)
            loss_mask=self.weight['mask']*self.mask_loss(mask_preds,mask_targets)
        loss_vmap = self.weight['vmap']*self.vmap_loss(vmap_preds, vmap_targets.unsqueeze(1))
        if self.add_edge:
            loss_edge = self.weight['edge']*self.vmap_loss(edge_preds, edge_targets.unsqueeze(1))
            loss['loss_edge'] = loss_edge
        
        loss_voff = self.weight['voff']*sigmoid_l1_loss(voff_preds,voff_targets,mask=vmap_targets.unsqueeze(1))
        loss['loss_mask'] = loss_mask
        loss['loss_vmap'] = loss_vmap
        loss['loss_voff'] = loss_voff
        return loss
    def predict_by_feat(self,
                        mask_preds: Tuple[Tensor],
                        results_list: List[InstanceData],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: ConfigDict,
                        rescale: bool = False,
                        activate_map: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        mask results.

        Args:
            mask_preds (tuple[Tensor]): Tuple of predicted foreground masks,
                each has shape (n, num_classes, h, w).
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert len(mask_preds) == len(results_list) == len(batch_img_metas)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            if results is None or len(results)==0:#在FCNMaskHead predict_by_feat基础上加的
                continue
            bboxes = results.rois#mask_head的roi，为预测bbox expand后的结果
            if bboxes.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='mask',
                    instance_results=[results],
                    mask_thr_binary=rcnn_test_cfg.mask_thr_binary)[0]
            else:
                im_mask = self._predict_by_feat_single(
                    mask_preds=mask_preds[img_id],
                    bboxes=bboxes,
                    labels=results.labels,
                    img_meta=img_meta,
                    rcnn_test_cfg=rcnn_test_cfg,
                    rescale=rescale,
                    activate_map=activate_map)
                results.masks = im_mask
        return results_list
def show_roi(rois_s,vmaps,pos_gt_inds,vmap_targets,voff_targets,save_dir='show/vmap_voff/'):
    """
    rois_s: tensor[n,4]
    vmaps: ndarray [m,256,256]
    pos_gt_inds: tensor[n]
    vmap_targets: tensor[n,96,96]
    voff_targets: tensor[n,2,96,96]
    """
    os.makedirs(save_dir,exist_ok=True)
    n=len(vmaps)
    ids=range(n)
    # random.sample(range(len(pos_gt_inds)),n)
    exist=len(os.listdir(save_dir))
    for k,i in enumerate(ids):
        roi=rois_s[i]
        vmap=vmaps[pos_gt_inds[i]]
        vmap_roi=vmap_targets[i]
        voff_roi=voff_targets[i]
        # Original vertex map
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(vmap)
        # Show ROI
        bbox = roi.numpy()
        rectangle = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rectangle)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f'{save_dir}/{exist+k}_original.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # RoI vertex map
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(vmap_roi)
        thr = 0.7
        select = vmap_roi > thr
        idx = np.where(select)
        x, y = idx[1], idx[0]  # vmap_gt's x, y coordinates
        plot_offsets(voff_roi[:, select], x, y, ax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f'{save_dir}/{exist+k}_roi.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved original vertex map {exist+k}.png")
    
def plot_offsets(offs, x, y, ax):
    """
    绘制偏移向量，用箭头表示。

    参数:
    offs : ndarray(2,n) 代表n个点处偏移向量的x,y分量
    x : ndarray(n) 代表n个点的x坐标
    y : ndarray(n) 代表n个点的y坐标
    ax : matplotlib轴对象，用于绘制
    """
    u, v = offs  # 偏移向量的x和y分量
    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=0.1,color='red',
              headwidth=3, headlength=2, headaxislength=2)#长度等于矢量长度除以scale
    ax.set_aspect('equal')