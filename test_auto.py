import argparse
import os
from mmengine.config import Config
from mmpl.engine.runner import PLRunner
import os.path as osp
from mmpl.registry import RUNNERS
from mmpl.utils import register_all_modules
register_all_modules()
def parse_args():
    parser = argparse.ArgumentParser(description='Train a pl model')
    parser.add_argument('--config', default='configs/auto_spacenet.py', help='train config file path')
    parser.add_argument('--ckpt_path', default='auto_spacenet.pth',help='checkpoint path')
    parser.add_argument('--status', default='test', help='fit or test', choices=['fit', 'test', 'predict', 'validate'])
    parser.add_argument('--work_dir', default='work_dir', help='the dir to save logs and mmpl')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    result_pth=f'{args.work_dir}/{cfg.task_name}/results.json'
    cfg.model_cfg.hyperparameters.evaluator=dict(
            test_evaluator=dict(type='CocoMetric',
                gt_path=f'dataset/spacenet/test/ann.json',
                result_pth=result_pth,
                metric_file=result_pth.replace(os.path.basename(result_pth),'metric.csv'),
                result_type=['mask','polygon'])
        )
    max_per_img=100
    cfg.model_cfg.panoptic_head.test_cfg.rcnn.max_per_img=max_per_img
    cfg.model_cfg.panoptic_head.roi_head.multi_process=True
    cfg.model_cfg.backbone.checkpoint=None
    cfg.callbacks=dict(
        type='DetVisualizationHook',
        draw=True,
        interval=1,
        score_thr=0.1,
        show=False,
        wait_time=1.,
        test_out_dir='visualization')
    cfg.trainer_cfg['logger'] = None
    cfg.trainer_cfg.default_root_dir=f'{args.work_dir}/debug/'
    if 'test_loader' not in cfg.datamodule_cfg:
        cfg.datamodule_cfg.test_loader = cfg.datamodule_cfg.val_loader
    
    cfg.trainer_cfg.devices=[args.gpu]
    cfg.trainer_cfg.use_distributed_sampler=False
    if args.work_dir is not None:
        cfg.trainer_cfg['default_root_dir'] = args.work_dir
    elif cfg.trainer_cfg.get('default_root_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.trainer_cfg['default_root_dir'] = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if 'runner_type' not in cfg:
        runner = PLRunner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.run(args.status, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()

