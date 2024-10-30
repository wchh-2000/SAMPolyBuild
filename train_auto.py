import argparse
import torch
import numpy as np
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
np.random.seed(2024)
from mmengine.config import Config
from mmpl.engine.runner import PLRunner
from mmpl.registry import RUNNERS
from mmpl.utils import register_all_modules
register_all_modules()
def parse_args():
    parser = argparse.ArgumentParser(description='Train a pl model')
    parser.add_argument('--config', default='configs/auto_spacenet.py',
                        help='train config file path')
    parser.add_argument('--is-debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--ckpt-path', default=None, help='checkpoint path')
    parser.add_argument('--status', default='fit', help='fit or test', choices=['fit', 'test', 'predict'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.is_debug:
        cfg.trainer_cfg['fast_dev_run'] = True
        cfg.trainer_cfg['logger'] = None
    if 'runner_type' not in cfg:
        runner = PLRunner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.run(args.status, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()

