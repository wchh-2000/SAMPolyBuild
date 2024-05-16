from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .metrics.polis import PolisEval
from .metrics.iou import iou_from_poly
from .metrics.boundaryF import boundaryF_from_poly
from .metrics.juncs_eval import precision_recall_from_vertex_set
import copy
import os,json
from shapely.geometry import Polygon
def Eval(annFile, resFile,type=['coco'],merge_sm=True):
    #merge_sm:是否合并small和medium
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
    cocoGt0,cocoDt0 = copy.deepcopy(cocoGt),copy.deepcopy(cocoDt)#coco评估后会转为mask，所以要保存原始的
    if 'coco' in type:
        imgIds = cocoGt.getImgIds()
        imgIds = imgIds[:]

        cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
        cocoEval.params.imgIds = imgIds
        cocoEval.params.catIds = [1]
        if merge_sm:
            cocoEval.params.areaRng = [[0,1e5 ** 2],[0,7500], [7500, 1e5 ** 2]]
            cocoEval.params.areaRngLbl = ['all','s&m', 'large']
        else:
            cocoEval.params.areaRng = [[0,1e5 ** 2],[0,32**2], [32**2, 7500], [7500, 1e5 ** 2]]
            cocoEval.params.areaRngLbl = ['all','small','medium', 'large']
        cocoEval.evaluate()
        cocoEval.accumulate()
        print('metric coco:')
        cocoEval.summarize(merge_sm=merge_sm)
    
    if 'polis' in type:
        print('metric polis:')
        polisEval = PolisEval(cocoGt0, cocoDt0)
        polisEval.evaluate()

def load_args(parser,path=None):
    args = parser.parse_args()
    if path is None:
        path=f'{args.work_dir}/{args.task_name}/args.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if key!='checkpoint' and key!='gpus' and key in vars(args):
                    setattr(args, key, value)
    return args
def load_train_args(parser):
    args = parser.parse_args()
    if 'config' in args:
        with open(args.config, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if key in vars(args):
                    setattr(args, key, value)
    return args

class PolygonMetrics:
    def __init__(self, divide_by_area=False, area_thresholds={'large': 7500, 'medium': 32**2}):
        self.divide_by_area = divide_by_area
        self.area_thresholds = area_thresholds

        self.metrics = {'precision': 0, 'recall': 0, 'miou': 0, 'vf1': 0, 'bound_f': 0}
        self.large_metrics = {'precision': 0, 'recall': 0, 'miou': 0, 'vf1': 0, 'bound_f': 0}
        self.medium_metrics = {'precision': 0, 'recall': 0, 'miou': 0, 'vf1': 0, 'bound_f': 0}
        self.small_metrics = {'precision': 0, 'recall': 0, 'miou': 0, 'vf1': 0, 'bound_f': 0}

        # self.count = 0
        self.large_count = 0
        self.medium_count = 0
        self.small_count = 0

    def calculate_metrics(self, pred_polygon, gt_polygon):
        p, r = precision_recall_from_vertex_set(pred_polygon[:-1, :], gt_polygon[:-1, :])#最后一个点为重复点，去除
        vf1 = 2 * p * r / (p + r + 1e-8)
        miou = iou_from_poly(pred_polygon, gt_polygon, 650, 650)#first to mask. todo 图像尺寸适配其他数据集 目前spacenet
        f, p_bound, r_bound = boundaryF_from_poly(pred_polygon, gt_polygon, 650, 650)

        if self.divide_by_area:
            area = Polygon(gt_polygon).area
            if area >= self.area_thresholds['large']:
                self.update_metrics(self.large_metrics, p, r, miou, vf1, f)
                self.large_count += 1
            elif area >= self.area_thresholds['medium']:
                self.update_metrics(self.medium_metrics, p, r, miou, vf1, f)
                self.medium_count += 1
            else:
                self.update_metrics(self.small_metrics, p, r, miou, vf1, f)
                self.small_count += 1
        else:
            self.update_metrics(self.metrics, p, r, miou, vf1, f)
            # self.count += 1

    def update_metrics(self, metrics, p, r, miou, f1, f):
        metrics['precision'] += p
        metrics['recall'] += r
        metrics['miou'] += miou
        metrics['vf1'] += f1
        metrics['bound_f'] += f

    def average_metrics(self, metrics, count):
        if count > 0:
            for key in metrics:
                metrics[key] /= count

    def compute_average(self,n=None):
        if self.divide_by_area:
            self.average_metrics(self.large_metrics, self.large_count)
            self.average_metrics(self.medium_metrics, self.medium_count)
            self.average_metrics(self.small_metrics, self.small_count)
            l,m,s = self.large_metrics.copy(), self.medium_metrics.copy(), self.small_metrics.copy()
            self.large_metrics = {'precision': 0, 'recall': 0, 'miou': 0, 'vf1': 0, 'bound_f': 0}#训练的一个validation_step清零，不然会累加
            self.medium_metrics = {'precision': 0, 'recall': 0, 'miou': 0, 'vf1': 0, 'bound_f': 0}
            self.small_metrics = {'precision': 0, 'recall': 0, 'miou': 0, 'vf1': 0, 'bound_f': 0}
            # self.count = 0
            self.large_count = 0
            self.medium_count = 0
            self.small_count = 0
            return l,m,s
        else:
            #divide_by_area=False时必传入n。训练的验证时为每个batch数，测试时为所有实例数n
            self.average_metrics(self.metrics, n)#no mask分数记为0，总数不排除no mask
            m=self.metrics.copy()
            self.metrics = {'precision': 0, 'recall': 0, 'miou': 0, 'vf1': 0, 'bound_f': 0}
            return m
