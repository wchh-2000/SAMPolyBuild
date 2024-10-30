import torch
from typing import Any

from mmpl.registry import MODELS
from .base_pler import BasePLer
from segment_anything import build_sam

import numpy as np
import time
def poly2bbox(poly):
    """
    input: (n, 2)
    output: (4)
    """
    x_min = np.min(poly[:,0])
    x_max = np.max(poly[:,0])
    y_min = np.min(poly[:,1])
    y_max = np.max(poly[:,1])
    return np.array([x_min, y_min, x_max-x_min, y_max-y_min]).tolist()
import cv2
def iou_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0
def iou_scores(polygons, masks):
    # 计算预测的polygons和masks的iou作为得分
    #polygons: list of (n, 2) ndarray
    #masks: (m, w, h) ndarray
    iou_scores = []
    for polygon, mask in zip(polygons, masks):
        height, width=mask.shape
        polygon_mask = cv2.fillPoly(np.zeros((height, width), dtype=np.uint8), [polygon.astype(np.int32)], 1)
        iou = iou_score(polygon_mask, mask)
        iou_scores.append(iou)
    return np.array(iou_scores)
@MODELS.register_module()
class SegSAMAnchorPLer(BasePLer):
    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 need_train_names=None,
                 train_cfg=None,
                 test_cfg=None,
                 chkpt=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.need_train_names = need_train_names

        self.backbone = build_sam(**backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self.panoptic_head = MODELS.build(panoptic_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.results=[]#for predict
        self.chkpt=chkpt

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.need_train_names is not None:
            self._set_grad(self.need_train_names, noneed_train_names=[])
        self.init_weights()

    def init_weights(self):
        if self.chkpt is not None:
            self.load_state_dict(torch.load(self.chkpt)['state_dict'])
            print('load weights from', self.chkpt)

    def train(self, mode=True):
        if self.need_train_names is not None:
            return self._set_train_module(mode, self.need_train_names)
        else:
            super().train(mode)
            return self

    @torch.no_grad()
    def extract_feat(self, batch_inputs):
        feat, inter_features = self.backbone.image_encoder(batch_inputs)
        return feat, inter_features

    def validation_step(self, batch, batch_idx):
        data = self.data_preprocessor(batch, False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']

        x = self.extract_feat(batch_inputs)
        # x = (
        # torch.rand(2, 256, 64, 64).to(self.device), [torch.rand(2, 64, 64, 768).to(self.device) for _ in range(12)])
        results = self.panoptic_head.predict(
            x, batch_data_samples, self.backbone)
        self.val_evaluator.process(batch, results)

    def training_step(self, batch, batch_idx):
        data = self.data_preprocessor(batch, True)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']
        x = self.extract_feat(batch_inputs)
        # x = (torch.rand(2, 256, 64, 64).to(self.device), [torch.rand(2, 64, 64, 768).to(self.device) for _ in range(12)])
        losses = self.panoptic_head.loss(x, batch_data_samples, self.backbone)

        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'train/{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses
        self.log_dict(log_vars, prog_bar=True)
        return log_vars
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        data = self.data_preprocessor(batch, False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']

        x = self.extract_feat(batch_inputs)
        # x = (
        # torch.rand(2, 256, 64, 64).to(self.device), [torch.rand(2, 64, 64, 768).to(self.device) for _ in range(12)])
        results = self.panoptic_head.predict(#mmpl/models/heads/sam_instance_head.py SAMAnchorInstanceHead
            x, batch_data_samples, self.backbone)
        self.predict_evaluator.update(batch, results)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # if batch_idx<529 or batch_idx>535:
        #     return
        data = self.data_preprocessor(batch, False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']
        # if batch_idx>5:
        #     return
        s=time.perf_counter()
        x = self.extract_feat(batch_inputs)
        # x = (
        # torch.rand(2, 256, 64, 64).to(self.device), [torch.rand(2, 64, 64, 768).to(self.device) for _ in range(12)])
        results = self.panoptic_head.predict(
            x, batch_data_samples, self.backbone)
        self.total_time+=(time.perf_counter()-s)
        self.test_evaluator.update(batch, results)
    def forward(self, inputs,batch_data_samples):
        x = self.extract_feat(inputs)
        results = self.panoptic_head.predict(
            x, batch_data_samples, self.backbone)
        return results





