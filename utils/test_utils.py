from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.metrics.polis import PolisEval
import copy
import os,json
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