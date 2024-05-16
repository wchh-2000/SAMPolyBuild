import os,json
join = os.path.join
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
#self defined:
from model import PromptModel
from utils.test_utils import load_args
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='prompt_instance_spacenet')
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--gpus',type=int, nargs='+', default=[0])
#model config:
parser.add_argument('--model_type', type=str, default='vit_b',help='for image encoder')
parser.add_argument('--freeze_img', type=bool, default=False,help='whether to freeze image encoder weights')
parser.add_argument('--freeze_mask', type=bool, default=False,help='whether to freeze mask decoder weights')
parser.add_argument('--upconv',type=bool,default=True,help='whether to use upsample and conv to upsample vmap voff')
parser.add_argument('--multi_mask',type=bool,default=True,help='whether to predict multi-mask')

parser.add_argument('--image_size', type=int, default=224,help='input image size to the model, a multiple of 16')#1024 for full img input
parser.add_argument('--instance_input',type=bool,default=True,help='whether to use region-focused strategy that input the instance area to the model')
parser.add_argument('--predict_crop_input',action='store_true',
                    help='whether to crop the image on-the-fly, instead of pre-cropping and saving.')
parser.add_argument('--img_dir', type=str, default='dataset/spacenet/test/images',help='image directory for the uncropped images')
parser.add_argument('--box_file', type=str, default='dataset/spacenet/test/ann.json',help='coco annotation file containing bounding boxes')
#data config:
parser.add_argument('--dataset', type=str, default='spacenet')
parser.add_argument('--ann_num', type=int, default=40,help="max annotation number for each training sample")#for full img input
parser.add_argument('--add_edge',type=bool,default=True,help='whether to add edge(boundary) prediction')
parser.add_argument('--gaussian', type=bool, default=True,help='whether to use gaussian kernel to generate vertex confidence map')
#prompt config:
parser.add_argument('--bbox', type=bool, default=True,help='whether to use bbox as prompt')
parser.add_argument('--mix_bbox_point', type=bool, default=True,help='whether to mix bbox and center point or multi-point prompts')
#post process config:
parser.add_argument('--max_distance', type=int, default=12)

#Load the same arguments from the training arguments (ignore the 'gpus' argument):
args = load_args(parser)

args.result_pth=f'{args.work_dir}/{args.task_name}/'
args.checkpoint=f'{args.work_dir}/{args.task_name}/version_0/checkpoints/bestIoU.ckpt'
args.ann_num=80
args_dict = vars(args)
print(args_dict)
os.makedirs(args.result_pth, exist_ok=True)

if args.predict_crop_input:
    eval=False
else:
    eval=True

#set Dataset:
dataset_param=dict(anns_per_sample=args.ann_num,input_size=args.image_size,
                   bbox=args.bbox,mix_bbox_point=args.mix_bbox_point,#prompt type
                   add_edge=args.add_edge,gaussian=args.gaussian)
if args.instance_input:
    if args.predict_crop_input:
        from dataset.dataset_crop import PromptDataset,collate_fn_test
        dataset = PromptDataset(args.img_dir, args.box_file,input_size=args.image_size)
        batch_size=1
        num_workers=2
    else:
        from dataset.dataset_pre_crop import PromptDataset,collate_fn_test
        dataset_pth = dict(data_root=f'dataset/{args.dataset}/test', ann_file='cropped.json', img_dir='cropped_images')
        dataset = PromptDataset(dataset_pth, 'test',**dataset_param)
        batch_size=60
        num_workers=6
else:#full img input
    from dataset.dataset_full_img import PromptDataset,collate_fn_test
    dataset_pth = dict(data_root=f'dataset/{args.dataset}/test', ann_file='ann.json', img_dir='images')
    dataset = PromptDataset(dataset_pth, 'test',**dataset_param)
    batch_size=1
    num_workers=2
dataloader = DataLoader(dataset, batch_size=batch_size,num_workers=num_workers, shuffle=False,
                    collate_fn=collate_fn_test,pin_memory=True)

class TestConfig:
    def __init__(self):
        self.train=False
        self.eval=eval
        self.log=False
        self.save_results=True
test_cfg=TestConfig()
device = 'cuda:'+str(args.gpus[0])

divide_by_area=True
model=PromptModel(args,test_cfg=test_cfg,divide_by_area=divide_by_area).to(device)
model.eval()
for step, batch in enumerate(tqdm(dataloader)):
    batch=model.transfer_batch_to_device(batch,device,step)
    model.validation_step(batch, step,log=False)
if eval:
    if divide_by_area:
        l_metrics, m_metrics, s_metrics = model.metrics_calculator.compute_average()
        #large medium small
        # 打开CSV文件
        csv_file_path = join(args.result_pth, f'metrics.csv')
        file_exists = os.path.isfile(csv_file_path)
        
        with open(csv_file_path, 'a', newline='') as csvfile:
            fieldnames = [
                'max_distance', 'miou_l', 'miou_m', 'miou_s', 
                'vf1_l', 'vf1_m', 'vf1_s',
                'bf1_l', 'bf1_m', 'bf1_s'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 如果文件不存在，则写入表头
            if not file_exists:
                writer.writeheader()
            # task = args.task_name.split('/')[1]
            writer.writerow({
                # 'task': task,
                'max_distance': args.max_distance,
                'miou_l': round(l_metrics['miou'] * 100, 2),
                'miou_m': round(m_metrics['miou'] * 100, 2),
                'miou_s': round(s_metrics['miou'] * 100, 2),
                'vf1_l': round(l_metrics['vf1'] * 100, 2),
                'vf1_m': round(m_metrics['vf1'] * 100, 2),
                'vf1_s': round(s_metrics['vf1'] * 100, 2),
                'bf1_l': round(l_metrics['bound_f'] * 100, 2),
                'bf1_m': round(m_metrics['bound_f'] * 100, 2),
                'bf1_s': round(s_metrics['bound_f'] * 100, 2)
            })
    else:
        metrics = model.metrics_calculator.compute_average(len(dataset))
        print(metrics)
    
if test_cfg.save_results:
    name=f'results_polyon_d{args.max_distance}.json'
    dt_file=join(args.result_pth,name)    
    with open(dt_file,'w') as _out:
        json.dump(model.results_poly,_out)
    print("Polygon results save to:",dt_file)#coco format