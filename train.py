import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse,json,os

from model import PromptModel
from utils.test_utils import load_train_args
join = os.path.join
torch.manual_seed(2023)
np.random.seed(2023)
debug=False
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='prompt_instance_spacenet')
#training config:
parser.add_argument('--config', type=str, default='configs/prompt_fullimg_spacenet.json',
                    help='in func load_train_args, use config file to update args')
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--gpus',type=int, nargs='+', default=[0])#[0,1]
parser.add_argument('--distributed', action='store_true',help='default False')
parser.add_argument('--epochs', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--batch_size_val', type=int, default=60)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--img_encoder_lr', type=float, default=5e-6)
parser.add_argument('--decoder_lr', type=float, default=5e-5)
parser.add_argument('--lr_step', type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
#model config:
parser.add_argument('--model_type', type=str, default='vit_b',help='for image encoder')
parser.add_argument('--checkpoint', type=str, default='segment_anything/sam_vit_b_01ec64.pth')
parser.add_argument('--freeze_img', type=bool, default=False,help='whether to freeze image encoder weights')
parser.add_argument('--freeze_mask', type=bool, default=False,help='whether to freeze mask decoder weights')
parser.add_argument('--upconv',type=bool,default=True,help='whether to use upsample and conv to upsample vmap voff')
parser.add_argument('--multi_mask',type=bool,default=True,help='whether to predict multi-mask')

parser.add_argument('--image_size', type=int, default=224,help='input image size to the model, a multiple of 16')#1024 for full img input
parser.add_argument('--instance_input',type=bool,default=True,
                    help='whether to use region-focused strategy that input the instance area to the model')
#data config:
parser.add_argument('--dataset', type=str, default='spacenet')
parser.add_argument('--ann_num', type=int, default=40,help="max annotation number for each training sample")#for full img input
parser.add_argument('--add_edge',type=bool,default=True,help='whether to add edge(boundary) prediction')
parser.add_argument('--gaussian', type=bool, default=True,help='whether to use gaussian kernel to generate vertex confidence map')
#prompt config:
parser.add_argument('--bbox', type=bool, default=True,help='whether to use bbox as prompt')
parser.add_argument('--mix_bbox_point', type=bool, default=True,help='whether to mix bbox and center point or multi-point prompts')
#post process config:
parser.add_argument('--max_distance', type=int, default=12,
                    help='max distance(T_dist in paper) to retain vertex in mask guided vertex connect')

args = load_train_args(parser)
if debug:
    args.task_name='debug'
if args.distributed:
    args.img_encoder_lr*=len(args.gpus)
    args.decoder_lr*=len(args.gpus)

args.log_dir = join(args.work_dir, args.task_name)
os.makedirs(args.log_dir, exist_ok=True)
args_dict = vars(args)
print(args_dict)
with open(join(args.log_dir,'args.json'), 'w') as f:
    json.dump(args_dict, f)

#set Dataset:
if args.instance_input:
    from dataset.dataset_pre_crop import PromptDataset,collate_fn_test
    train_dataset_pth = dict(data_root=f'dataset/{args.dataset}/train', ann_file='cropped.json', img_dir='cropped_images')
    val_dataset_pth = dict(data_root=f'dataset/{args.dataset}/val', ann_file='cropped.json', img_dir='cropped_images')
else:
    from dataset.dataset_full_img import PromptDataset,collate_fn_test
    args.batch_size=1
    args.batch_size_val=1
    args.num_workers=2
    train_dataset_pth = dict(data_root=f'dataset/{args.dataset}/train', ann_file='ann.json', img_dir='images')
    val_dataset_pth = dict(data_root=f'dataset/{args.dataset}/val', ann_file='ann.json', img_dir='images')
dataset_param=dict(anns_per_sample=args.ann_num,input_size=args.image_size,
                        add_edge=args.add_edge,gaussian=args.gaussian,
                        bbox=args.bbox,mix_bbox_point=args.mix_bbox_point)
train_dataset = PromptDataset(train_dataset_pth,**dataset_param)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.num_workers,
                                collate_fn=collate_fn_test,shuffle=True,pin_memory=True)
val_dataset = PromptDataset(val_dataset_pth, mode='val',**dataset_param)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_val,num_workers=args.num_workers,
                            collate_fn=collate_fn_test, shuffle=False,pin_memory=True)

class TestConfig:
    def __init__(self):
        self.train=True
        self.eval=True
        self.save_results=False
        self.log=True
test_cfg=TestConfig()
model = PromptModel(args, test_cfg=test_cfg)
if args.distributed:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

checkpoint_callback = ModelCheckpoint(
        monitor='val/mIoU',
        mode='max',
        every_n_epochs=1,
        save_last=True,
        save_top_k=1,
        filename='bestIoU'
        )
logger = TensorBoardLogger(args.work_dir, name=args.task_name)
train_param=dict(
    max_epochs=args.epochs,
    log_every_n_steps=50,
    devices=args.gpus,
    # accelerator='cpu',
    logger=logger,
    default_root_dir=args.log_dir,
    callbacks=[checkpoint_callback])
if args.distributed:
    train_param.update(dict(
        accelerator="gpu", strategy="ddp_find_unused_parameters_true"))
if debug:
    train_param.update(dict(
        limit_train_batches=10,
        limit_val_batches=10
        ))
trainer = pl.Trainer(**train_param)
trainer.fit(model, train_dataloader, val_dataloader)
