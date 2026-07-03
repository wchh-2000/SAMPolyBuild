import argparse
from pathlib import Path

from mmengine.config import Config
from mmpl.engine.runner import PLRunner
from mmpl.registry import RUNNERS
from mmpl.utils import register_all_modules
register_all_modules()
def parse_args():
    parser = argparse.ArgumentParser(description='Train a pl model')
    parser.add_argument('--config', default='configs/auto_whumix.py', help='train config file path')
    parser.add_argument('--ckpt_path', default='auto_whumix.pth',help='checkpoint path')
    parser.add_argument('--status', default='predict', help='fit or test', choices=['fit', 'test', 'predict', 'validate'])
    parser.add_argument('--work_dir', default='work_dir', help='the dir to save logs and mmpl')
    parser.add_argument('--img_dir', default='dataset/whu_mix/test2/images/')
    parser.add_argument('--img_suffix', default='.tif')
    parser.add_argument('--score_thr', default=0.1, type=float, help='score threshold')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=6, type=int, help='predict batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='predict dataloader workers')
    args = parser.parse_args()
    return args


def run_auto_inference(
        config,
        ckpt_path,
        img_dir,
        work_dir='work_dir',
        img_suffix='.tif',
        score_thr=0.1,
        gpu=0,
        batch_size=6,
        num_workers=2,
        status='predict',
        result_dir=None,
        replace_predict_loader=False):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')
    if num_workers < 0:
        raise ValueError('num_workers must be non-negative.')

    cfg = Config.fromfile(config)
    base_work_dir = (
        Path(work_dir)
        if work_dir is not None
        else Path('./work_dirs') / Path(config).stem
    )
    result_dir = Path(result_dir or base_work_dir / cfg.task_name)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_pth = str(result_dir / 'results.json')
    results_mask_pth = result_pth.replace('results', 'results_mask')
    Path(results_mask_pth).parent.mkdir(parents=True, exist_ok=True)
    cfg.model_cfg.hyperparameters.evaluator=dict(predict_evaluator=
                                                 dict(type='CocoMetric',
                                            result_pth=result_pth,
                                            evaluate=False,
                                            score_thr=score_thr,
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
    if replace_predict_loader or 'predict_loader' not in cfg.datamodule_cfg:
        cfg.datamodule_cfg.predict_loader = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            dataset=dict(
                type='PredictDataset',
                data_root=str(img_dir),
                # data_prefix=dict(img_path=''),
                img_suffix=img_suffix,
                pipeline=[
                    dict(type='mmdet.LoadImageFromFile', backend_args=None),
                    dict(type='mmdet.Resize', scale=(1024, 1024)),
                    dict(
                        type='mmdet.PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                'scale_factor'))
                ],
                backend_args=None))
    cfg.trainer_cfg.devices=[gpu]
    cfg.trainer_cfg.use_distributed_sampler=False
    cfg.trainer_cfg['default_root_dir'] = str(base_work_dir)

    if 'runner_type' not in cfg:
        runner = PLRunner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.run(status, ckpt_path=str(ckpt_path))
    return {
        'task_name': cfg.task_name,
        'results_path': result_pth,
        'results_mask_path': results_mask_pth,
    }


def main():
    args = parse_args()
    run_auto_inference(
        config=args.config,
        ckpt_path=args.ckpt_path,
        img_dir=args.img_dir,
        work_dir=args.work_dir,
        img_suffix=args.img_suffix,
        score_thr=args.score_thr,
        gpu=args.gpu,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        status=args.status,
    )


if __name__ == '__main__':
    main()

