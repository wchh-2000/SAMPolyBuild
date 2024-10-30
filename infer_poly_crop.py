#%%
from segment_anything import build_sam, SamPredictor
import numpy as np
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
join = os.path.join
import argparse
import matplotlib.pyplot as plt
#self defined:
from utils.test_utils import load_args
from utils.post_process import GetPolygons
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='yellow', facecolor=(0,0,0,0), lw=2))    
def show_points(coords, labels, ax,marker_size=100, label=''):
    
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                s=marker_size, edgecolor='white', linewidth=1.25,label=label)
    if neg_points.shape[0] > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
# set up the parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='prompt_instance_spacenet')
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--imgpth', type=str, default='figs/eg.jpg')
parser.add_argument('--gpu',type=int, default=0)
#model config:
parser.add_argument('--model_type', type=str, default='vit_b',help='for image encoder')
parser.add_argument('--upconv',type=bool,default=True,help='whether to use upsample and conv to upsample vmap voff')
parser.add_argument('--multi_mask',type=bool,default=True,help='whether to predict multi-mask')

parser.add_argument('--image_size', type=int, default=224,help='input image size to the model, a multiple of 16')#1024 for full img input
parser.add_argument('--instance_input',type=bool,default=True,help='whether to use region-focused strategy that input the instance area to the model')
parser.add_argument('--add_edge',type=bool,default=True,help='whether to add edge(boundary) prediction')
parser.add_argument('--gaussian', type=bool, default=True,help='whether to use gaussian kernel to generate vertex confidence map')
parser.add_argument('--max_distance', type=int, default=10)

args = load_args(parser,path='configs/prompt_instance_spacenet.json')
args.result_pth=f'{args.work_dir}/{args.task_name}/'
args.checkpoint='prompt_interactive.pth'#f'{args.work_dir}/{args.task_name}/version_0/checkpoints/bestIoU.ckpt'
os.makedirs(args.result_pth,exist_ok=True)
print(vars(args))
def get_prompt(bbox,imgsize=512,prompt_point=None):
    """
    输入相对原图的bbox(xmin,ymin,xmax,ymax),prompt_point (n,2)
    原图尺寸imgsize int
    输出相对原图的裁剪框和相对裁剪框的提示点坐标point和框坐标new_bbox
    """
    x0, y0, xmax, ymax = bbox
    w0,h0=xmax-x0,ymax-y0
    x=x0-w0*0.075
    y=y0-h0*0.075
    w=w0*1.15
    h=h0*1.15
    #处理边界：
    x=int(max(0,x))
    y=int(max(0,y))
    w=int(min(w,imgsize-x))
    h=int(min(h,imgsize-y))
    bbox_crop=[x,y,x+w,y+h]
    #裁剪区域左上角坐标和x,y轴的尺度因子:
    scale_factor_x, scale_factor_y = 1,1#w/args.image_size, h/args.image_size
    #args.image_size为实例区域统一尺寸，用于输入模型
    # pos_transform=[x,y,scale_factor_x,scale_factor_y]#位置坐标变换参数
    new_bbox = [
        (x0 - x) / scale_factor_x,
        (y0 - y) / scale_factor_y,
        w0 / scale_factor_x,
        h0 / scale_factor_y
    ]
    point = np.array([new_bbox[0] + new_bbox[2] / 2, new_bbox[1] + new_bbox[3] / 2]).reshape(1,2)
    if prompt_point is not None:
        prompt_point = (prompt_point - [x, y]) / [scale_factor_x, scale_factor_y]
        point = np.concatenate([point, prompt_point], axis=0)
    #x,y,w,h->x1,y1,x2,y2
    new_bbox[2] = new_bbox[0] + new_bbox[2]
    new_bbox[3] = new_bbox[1] + new_bbox[3]
    return bbox_crop,point,np.array(new_bbox)
device = 'cuda:'+str(args.gpu)
sam_model = build_sam(use_poly=True,load_pl=True,**vars(args)).to(device)
sam_model.eval()
predictor = SamPredictor(sam_model,polygon=True)   
#%% load image
image = cv2.imread(args.imgpth)

# %% set prompt and predict:
bbox=[519,506,572,586]#左上角和右下角坐标xyxy
# no prompt point:
# prompt_point=None
# label = np.array([1])#0 for background point, 1 for instance point

# with prompt point:
prompt_point=np.array([[541,582]])
label = np.array([1,1])

bbox_crop,point,new_bbox=get_prompt(bbox,imgsize=image.shape[1],prompt_point=prompt_point)
image=image[bbox_crop[1]:bbox_crop[3],bbox_crop[0]:bbox_crop[2],:]
image = image[:, :, ::-1]  # BGR to RGB

predictor.set_image_resize(image)#含ResizeLongestSide到image_size,to_tensor
mask, score, logit,pred_poly = predictor.predict(
    point_coords=point,
    point_labels=label,
    box=new_bbox,
    multimask_output=args.multi_mask,
)
pred_vmap,pred_voff=pred_poly['vmap'],pred_poly['voff']
pred_vmap = torch.sigmoid(pred_vmap)
pred_voff=torch.sigmoid(pred_voff)
crop_w,crop_h=bbox_crop[2]-bbox_crop[0],bbox_crop[3]-bbox_crop[1]
# print(mask.shape,pred_vmap.shape,pred_voff.shape,crop_w,crop_h)
if args.multi_mask:
    mask=mask[np.argmax(score),:,:].reshape(1,crop_h,crop_w)
polygon, score, _=GetPolygons(mask,pred_vmap,pred_voff,ori_size=(crop_w,crop_h),
                              max_distance=args.max_distance)
polygon=polygon[0]
#可视化：
plt.figure() 
plt.imshow(image)
show_points(point,label, plt.gca(),marker_size=400)
if new_bbox is not None:
    show_box(new_bbox, plt.gca())
#show polygon:
plt.plot(polygon[:,0], polygon[:,1],color='r',linewidth=3)
plt.scatter(polygon[:,0], polygon[:,1],color='r',linewidths=3,marker='.')
plt.axis('off')
# plt.show()
plt.savefig(join(args.result_pth,'result.jpg'),bbox_inches='tight',pad_inches=0)