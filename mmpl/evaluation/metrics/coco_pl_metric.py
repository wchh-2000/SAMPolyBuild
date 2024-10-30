# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmdet.structures.mask import encode_mask_results
from mmdet.evaluation.functional import eval_recalls
from torchmetrics import Metric
from mmpl.registry import METRICS
from torchmetrics.utilities import rank_zero_info
from pycocotools import mask as coco_mask
from shapely.geometry import Polygon
import json,os
import csv
from utils.metrics.coco_IoU_all import compute_IoU
join=os.path.join
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
def xyxy2xywh(bbox: np.ndarray) -> list:
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.

    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.

    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """

    _bbox: List = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]
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
@METRICS.register_module()
class CocoMetric(Metric):#测试/推理predict时用
    def __init__(self,gt_path=None,result_pth=None,score_thr=0.1,
        result_type=['polygon','mask'],evaluate=True,merge_sm=True,metric_file=None):
        super().__init__()
        if gt_path:
            self.coco_api = COCO(gt_path)
        self.result_type=result_type
        self.result_pth=result_pth
        self.merge_sm=merge_sm #merge small and medium objects
        self.evaluate=evaluate
        self.metric_file=metric_file
        self.score_thr=score_thr
        
        # Initialize states
        if 'polygon' in result_type:
            self.add_state('results_poly', default=[], dist_reduce_fx=None)
        if 'mask' in result_type:
            self.add_state('results_mask', default=[], dist_reduce_fx=None)

    def update(self,data_batch: dict, data_samples: Sequence[dict]):
        #results: A batch of data samples that contain predictions.
        
        for data_sample in data_samples:
            pred = data_sample.pred_instances
            if len(pred)==0:#图中无结果的情况
                continue
            img_id = data_sample.img_id
            polygons=pred['polygons']
            bboxes=pred['bboxes'].cpu().numpy()
            bboxes=[xyxy2xywh(bbox) for bbox in bboxes]
            masks=pred['masks'].detach().cpu().numpy()
            scores=pred['scores'].cpu().numpy()
            scores_cls=scores.copy()
            mask_scores=pred['mask_scores']
            IoU_scores=iou_scores(polygons, masks)#计算预测的polygons和masks的iou作为得分
            scores=np.mean([scores,mask_scores,IoU_scores],axis=0)#三者平均
            # if len(polygons)!=len(masks) or len(polygons)!=len(scores):
            #     print(f"polygons and masks have different length in img {img_id}")
            if 'polygon' in self.result_type:
                # Preparing coco format for polygons:
                for polygon,bbox, score,score_cls in zip(polygons,bboxes, scores,scores_cls):
                    if score_cls < self.score_thr or polygon is None:
                        continue
                    vec_poly = polygon.ravel().tolist()
                    vec_poly = [round(x, 2) for x in vec_poly]
                    # poly_bbox = poly2bbox(polygon)
                    ann_poly = {
                        'image_id': img_id,
                        'category_id': 1,
                        'segmentation': [vec_poly],
                        'bbox': [round(bb,2) for bb in bbox],
                        'area': Polygon(polygon).area,#无面积时评价用bbox面积
                        'score': float(score),
                        'score_cls':float(score_cls)
                    }
                    self.results_poly.append(ann_poly)
            if 'mask' in self.result_type:
                # Preparing coco format for masks:
                for mask,bbox, score,score_cls in zip(masks,bboxes, scores,scores_cls):
                    if score_cls < self.score_thr:
                        continue
                    encoded_region = coco_mask.encode(np.asfortranarray(mask))
                    ann_mask = {
                        'image_id': img_id,
                        'category_id': 1,
                        'segmentation': {
                            "size": encoded_region["size"],
                            "counts": encoded_region["counts"].decode()
                        },
                        'bbox': bbox,
                        'score': float(score),
                        'score_cls':float(score_cls)
                    }
                    ann_mask['area']=int(coco_mask.area(ann_mask['segmentation']))#无面积时评价用bbox面积
                    self.results_mask.append(ann_mask)
    def compute(self):
        # Compute mask and polygon's coco metric using self.results_poly and self.results_mask
        if 'polygon' in self.result_type:
            if len(self.results_poly)==0:
                return {}
            if self.result_pth is not None:
                with open(self.result_pth,'w') as _out:
                    json.dump(list(self.results_poly),_out)
                print(f"polygon result saved to {self.result_pth}")
            if self.evaluate:
                # Evaluation for polygons
                pred_coco_poly = self.coco_api.loadRes(self.results_poly)
                coco_eval_poly = COCOeval(self.coco_api, pred_coco_poly, iouType='segm')
                if self.merge_sm:
                    coco_eval_poly.params.areaRng = [[0,1e5 ** 2],[0,7500], [7500, 1e5 ** 2]]
                    coco_eval_poly.params.areaRngLbl = ['all','s&m', 'large']
                else:
                    coco_eval_poly.params.areaRng = [[0,1e5 ** 2],[0,32**2], [32**2, 7500], [7500, 1e5 ** 2]]
                    coco_eval_poly.params.areaRngLbl = ['all','small','medium', 'large']
                coco_eval_poly.evaluate()
                coco_eval_poly.accumulate()
                print("polygon coco eval:")
                coco_eval_poly.summarize(merge_sm=self.merge_sm)
        if 'mask' in self.result_type:
            if len(self.results_mask)==0:
                return {}                
            if self.result_pth is not None:
                mask_pth=self.result_pth.replace('results','results_mask')
                with open(mask_pth,'w') as _out:
                    json.dump(list(self.results_mask),_out)
                print(f"mask result saved to {mask_pth}")
            if self.evaluate:
                # Evaluation for masks
                pred_coco_mask = self.coco_api.loadRes(self.results_mask)
                coco_eval_mask = COCOeval(self.coco_api, pred_coco_mask, iouType='segm')
                if self.merge_sm:
                    coco_eval_mask.params.areaRng = [[0,1e5 ** 2],[0,7500], [7500, 1e5 ** 2]]
                    coco_eval_mask.params.areaRngLbl = ['all','s&m', 'large']
                else:
                    coco_eval_mask.params.areaRng = [[0,1e5 ** 2],[0,32**2], [32**2, 7500], [7500, 1e5 ** 2]]
                    coco_eval_mask.params.areaRngLbl = ['all','small','medium', 'large']
                coco_eval_mask.evaluate()
                coco_eval_mask.accumulate()
                print("mask coco eval:")
                coco_eval_mask.summarize(merge_sm=self.merge_sm)
                iou=compute_IoU(self.coco_api, pred_coco_mask)/100
        if self.evaluate:
            if self.merge_sm:
                metrics = {
                    "AP": coco_eval_poly.stats[0],
                    "APs&m": coco_eval_poly.stats[3],
                    "APl": coco_eval_poly.stats[4],
                    "AR": coco_eval_poly.stats[5],
                    "ARs&m": coco_eval_poly.stats[6],
                    "ARl": coco_eval_poly.stats[7],
                }
                if 'mask' in self.result_type:
                    metrics.update({
                        "APmask": coco_eval_mask.stats[0],
                        "APmasks&m": coco_eval_mask.stats[3],
                        "APmaskl": coco_eval_mask.stats[4],
                        "IoU":iou
                    })
            else:
                metrics = {
                    "AP": coco_eval_poly.stats[0],
                    # "AP50": coco_eval_poly.stats[1],
                    # "AP75": coco_eval_poly.stats[2],
                    "APs": coco_eval_poly.stats[3],
                    "APm": coco_eval_poly.stats[4],
                    "APl": coco_eval_poly.stats[5],          
                    "AR": coco_eval_poly.stats[8],
                    "ARs": coco_eval_poly.stats[9],
                    "ARm": coco_eval_poly.stats[10],
                    "ARl": coco_eval_poly.stats[11],
                }
                if 'mask' in self.result_type:
                    metrics.update({
                    "APmask": coco_eval_mask.stats[0],
                    "APmasks": coco_eval_mask.stats[3],
                    "APmaskm": coco_eval_mask.stats[4],
                    "APmaskl": coco_eval_mask.stats[5]})
            if self.metric_file:
                with open(self.metric_file, 'a+', newline='') as _out:
                    writer = csv.writer(_out)
                    if _out.tell() == 0:  # Check if file is empty
                        header = list(metrics.keys())
                        writer.writerow(header)
                    row = [round(v * 100, 1) for v in metrics.values()]
                    writer.writerow(row)

            return metrics
        else:
            return None
from mmengine.evaluator import BaseMetric
@METRICS.register_module()
class CocoPLMetric(BaseMetric):#验证时用
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
    """
    # default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 cls_thr=0.2,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self._dataset_meta: Union[None, dict] = None
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast','mask','poly']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)
        self.cls_thr=cls_thr#分类分数阈值，小于的舍弃

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:#默认None
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None

        # self.add_state('results', default=[], dist_reduce_fx="cat")
        # self.add_state('results_mask', default=[], dist_reduce_fx=None)

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta

    def fast_eval_recall(self,
                         results: List[dict],
                         proposal_nums: Sequence[int],
                         iou_thrs: Sequence[float],
                         logger: Optional[MMLogger] = None) -> np.ndarray:
        """Evaluate proposal recall with COCO's fast_eval_recall.

        Args:
            results (List[dict]): Results of the dataset.
            proposal_nums (Sequence[int]): Proposal numbers used for
                evaluation.
            iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
            logger (MMLogger, optional): Logger used for logging the recall
                summary.
        Returns:
            np.ndarray: Averaged recall results.
        """
        gt_bboxes = []
        pred_bboxes = [result['bboxes'] for result in results]
        for i in range(len(self.img_ids)):
            ann_ids = self._coco_api.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self._coco_api.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, pred_bboxes, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        poly_json_results = []
        segm_json_results = []
        for idx, result in enumerate(results):
            if result is None:
                continue
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            bboxes = [self.xyxy2xywh(bbox) for bbox in bboxes]
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = bboxes[i]
                data['area'] = bboxes[i][2] * bboxes[i][3]
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)


            # segm results
            masks=result['masks']
            if len(masks)>0:
                scores_poly = result.get('scores_poly', scores)
                for i, label in enumerate(labels):
                    if scores[i]<self.cls_thr:
                        continue
                    data = dict()
                    data['image_id'] = image_id
                    data['bbox'] = bboxes[i]
                    data['score'] = float(scores_poly[i])
                    data['category_id'] = self.cat_ids[label]
                    if 'polygons' in result:
                        poly = result['polygons'][i]#多边形顶点列表[[x1,y1,x2,y2...]]
                        data['area']=Polygon(np.array(poly).reshape(-1,2)).area
                        data['segmentation'] = poly
                        poly_json_results.append(data)

                    data_mask=data.copy()
                    if isinstance(masks[i]['counts'], bytes):
                        masks[i]['counts'] = masks[i]['counts'].decode()
                    data_mask['segmentation'] = masks[i]
                    segm_json_results.append(data_mask)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if len(poly_json_results)>0:
            result_files['poly'] = f'{outfile_prefix}.poly.json'
            dump(poly_json_results, result_files['poly'])
        if len(segm_json_results)>0:
            result_files['mask'] = f'{outfile_prefix}.mask.json'
            dump(segm_json_results, result_files['mask'])
        return result_files

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:#data_samples[i].pred_instances.polygons 为第i个图像中的所有建筑物的多边形
            # parse gt
            gt = dict()
            gt['width'] = data_sample.ori_shape[1]
            gt['height'] = data_sample.ori_shape[0]
            gt['img_id'] = data_sample.img_id
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'gt_instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = []
                for x_data in data_sample.gt_instances:
                    mask_ = encode_mask_results(x_data['masks'].masks)
                    assert len(mask_) == 1, \
                        'Only support one mask per instance for now'
                    gt['anns'].append(
                        dict(
                            bbox_label=x_data['labels'].item(),
                            bbox=x_data['bboxes'].cpu().numpy().reshape(4),
                            mask=mask_[0]
                        )
                    )
            # parse prediction:        
            result = dict(masks=[], bboxes=[], scores=[], labels=[])
            pred = data_sample.pred_instances
            if len(pred)==0:
                self.results.append((gt, None))
                continue
            result['img_id'] = data_sample.img_id
            bboxes = pred['bboxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            masks=pred['masks'].detach().cpu().numpy()
            if 'polygons' in pred:
                mask_scores=pred['mask_scores']
                IoU_scores=iou_scores(pred['polygons'], masks)
                scores_poly=np.mean([scores,mask_scores,IoU_scores],axis=0)#三者平均
                result['scores_poly']=scores_poly
                result['polygons']=[[polygon.ravel().tolist()] for polygon in pred['polygons']]
            result['bboxes']=bboxes
            result['labels']=labels
            result['scores']=scores
            # encode mask to RLE:
            result['masks'] = encode_mask_results(masks)                    
            # some detectors use different scores for bbox and mask
            # if 'mask_scores' in pred:
            #     result['mask_scores'] = pred['mask_scores']

            
            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self,results) -> Dict[str, float]:#多卡验证时results为多卡gather的结果
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # results = self.results
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')#多卡验证时，每卡都有对应的gt和preds,最终结果平均
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.getCatIds()
        if self.img_ids is None:
            self.img_ids = self._coco_api.getImgIds()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results
        # mapping of cocoEval.stats
        self.coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@500': 8,
            'AR_s@500': 9,
            'AR_m@500': 10,
            'AR_l@500': 11
        }
        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'bbox' else 'segm'
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            # try:
            predictions = load(result_files[metric])
            # if iou_type == 'segm':
                # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                # When evaluating mask AP, if the results contain bbox,
                # cocoapi will use the box area instead of the mask area
                # for calculating the instance area. Though the overall AP
                # is not affected, this leads to different
                # small/medium/large mask AP results.
                # for x in predictions:
                #     x.pop('bbox')
            coco_dt = self._coco_api.loadRes(predictions)
            if metric =='mask':
                miou=compute_IoU(self._coco_api,coco_dt)
                eval_results['mask_IoU']=miou

            coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs
            coco_eval.params.areaRng = [[0,1e5 ** 2],[0,32**2], [32**2, 7500], [7500, 1e5 ** 2]]
            coco_eval.params.areaRngLbl = ['all','small','medium', 'large']

            
            metric_items = self.metric_items
            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_s', 'mAP_m', 'mAP_l','AR@500','AR_s@500','AR_l@500'
                ]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()     

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = coco_eval.stats[self.coco_metric_names[metric_item]]
                eval_results[key] = float(f'{round(val, 3)}')

            ap = coco_eval.stats[:6]
            # if mmengine.dist.get_local_rank() == 0:

            rank_zero_info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                        f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                        f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        # for k, v in eval_results.items():
        #     eval_results[k] = torch.tensor(v).to(self.device)
        eval_results={'val'+k:v for k,v in eval_results.items()}
        self._coco_api = None
        return eval_results
