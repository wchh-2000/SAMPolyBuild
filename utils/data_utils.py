import numpy as np
import cv2
from numpy.random import randint
from shapely.geometry import Polygon, box,GeometryCollection,MultiLineString,MultiPolygon,LineString
from pycocotools import mask as maskUtils
def get_bbox_point(ann, scale_rate=1, crop_bbox=None, hflip=False, vflip=False, jitter=True,
                   add_center=True,input_size=1024):
    # 几何变换：随机缩放，随机裁剪到crop_bbox，最终垂直/水平翻转，相应的bbox和point也要变换 todo: 修改crop_bbox hflip vflip部分
    # crop_bbox (x1, y1, x2, y2) 左上右下
    bbox = np.array(ann['bbox'])  # x, y, w, h
    #transform:
    if scale_rate != 1:
        bbox *= scale_rate
    w,h=bbox[2],bbox[3]
    if crop_bbox is not None:
        x1, y1, x2, y2 = crop_bbox #x2=x1+512,y2=y1+512
        # 保留bbox在crop_bbox内的部分，重新计算bbox在crop_bbox内的坐标:
        bbox[0] = max(bbox[0], x1)-x1
        bbox[1] = max(bbox[1], y1)-y1
        bbox[2] = min(w, 512-bbox[0])
        bbox[3] = min(h, 512-bbox[1])    
    if hflip:
        bbox[0]=512-bbox[0]-bbox[2] #注意bbox[0]还是左上点
    if vflip:
        bbox[1]=512-bbox[1]-bbox[3]
    #取新的bbox中心点:
    if add_center:
        point = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
    if jitter:
        # bbox坐标和point坐标都要加上随机扰动，扰动范围为bbox的w和h的1/10 w,h扰动范围1/20
        jitter_amount_bbox = np.array([bbox[2] / 10, bbox[3] / 10, bbox[2] / 20, bbox[3] / 20])
        bbox += np.random.uniform(-jitter_amount_bbox, jitter_amount_bbox)        
        if add_center:
            jitter_amount_point = np.array([bbox[2] / 10, bbox[3] / 10])            
            point += np.random.uniform(-jitter_amount_point, jitter_amount_point)
    #x,y,w,h->x1,y1,x2,y2
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
    bbox=np.clip(bbox,0,512-1e-4)#防止bbox出界
    if input_size!=512:
        bbox*=input_size/512
        #512->input_size
    if add_center:
        if input_size!=512:
            point*=input_size/512
        point=point.reshape(1,2)
        return bbox, point
    else:
        return bbox
def get_points_in_polygon(polygon,gt_size=256,input_size=1024):
    #随机生成多边形内的1至3个点坐标，且坐标在多边形内靠近边缘的位置的概率更大
    #polygons: np.array(n,2) x,y (w,h)    gt_w,gt_h（gt_size）范围内
    #返回在input_size*input_size(输入sam模型）范围内的坐标
    # Compute polygon centroid
    centroid = np.mean(polygon[:-1], axis=0)#去掉最后一个点，因为和第一个点重复
    pos_num = np.random.randint(1, 3)#随机生成1-2个点
    neg_num = np.random.randint(0, 3)#随机生成0-2个负样本点
    keypoints = [generate_point_near_edge(polygon,centroid) for _ in range(pos_num)]
    
    #坐标变到input_size*input_size范围内:
    keypoints=np.array(keypoints)*input_size/gt_size
    if neg_num>0:
        neg_keypoints=[]
        for _ in range(neg_num):
            neg=generate_point_outside_polygon(polygon,centroid,gt_size=gt_size)
            if neg is None:
                neg_num-=1
            else:
                neg_keypoints.append(neg)
        
    #关键点列表padding，加标签:
    points=np.zeros((4,2))#统一补齐成4个点 无效点用0填充
    points[:pos_num,:]=keypoints
    label=-np.ones(4)#无效标签用-1填充
    label[:pos_num]=np.ones(pos_num)#正样本标签用1填充
    if neg_num>0:
        neg_keypoints=np.array(neg_keypoints)*input_size/gt_size
        points[pos_num:pos_num+neg_num,:]=neg_keypoints
        label[pos_num:pos_num+neg_num]=np.zeros(neg_num)#负样本标签用0填充
    return points,label
#%% for get_points_in_polygon
import matplotlib.path as mpltPath
def is_point_inside_polygon(polygon, point):
    """Check if point is inside polygon."""
    path = mpltPath.Path(polygon)
    return path.contains_point(point)
def random_point_on_edge(polygon):
    """Generate a random point on a random edge of the polygon."""
    # Choose a random edge
    edge_index = np.random.randint(0, len(polygon) - 1)
    p1, p2 = polygon[edge_index], polygon[edge_index + 1]

    # Generate a random weight
    alpha = np.random.random()

    # Interpolate between the two points of the edge
    return [(1 - alpha) * p1[0] + alpha * p2[0], (1 - alpha) * p1[1] + alpha * p2[1]]

def generate_point_near_edge(polygon,centroid):
    """Generate a point near the edge of the polygon."""
    i=0
    while True:
        # Get random point on edge
        point = random_point_on_edge(polygon)

        # Move the point slightly towards the centroid
        move_ratio = np.random.uniform(0.07, 0.5)  # 靠近中心点的程度，中心点与边缘点距离为幅度1
        point = [(1 - move_ratio) * point[0] + move_ratio * centroid[0],
                 (1 - move_ratio) * point[1] + move_ratio * centroid[1]]

        # If point is inside the polygon, return it
        if is_point_inside_polygon(polygon, point):
            return point
        i+=1
        if i>3:#防止多次不在多边形内
            return centroid
def generate_point_outside_polygon(polygon,centroid,gt_size=256):
    """Generate a point outside the polygon along the line from centroid to a point on the edge."""
    # Get random point on edge   todo:增加超过图像边界判断 gt_size为边界
    i=0
    while True:
        point = random_point_on_edge(polygon)
        # Move the point beyond the edge, along the line connecting the centroid to the point on the edge
        move_ratio = np.random.uniform(0.01, 0.15)  #沿着中心点到边缘点的线向外移动的程度,中心点与边缘点距离为幅度1 （0.1，0.8）
        point = [(1 + move_ratio) * point[0] - move_ratio * centroid[0],
                (1 + move_ratio) * point[1] - move_ratio * centroid[1]]
        image_boundary=[(0,0),(0,gt_size),(gt_size,gt_size),(gt_size,0)]
        if is_point_inside_polygon(image_boundary, point):
            return point
        i+=1
        if i>3:
            return None

def get_point_ann(junctions,gt_h,gt_w,gaussian=True,sigma=1):
    #junctions: np.array(n,2) x,y (w,h) gt_w,gt_h(256*256)范围内 vmap, voff大小为gt_h,gt_w
    #获取顶点激活图vmap和偏移量图voff：
    vmap = np.zeros((gt_h, gt_w))
    voff = np.zeros((2, gt_h, gt_w))
    junctions[:,0]=np.clip(junctions[:,0],0,gt_w-1e-4)
    junctions[:,1]=np.clip(junctions[:,1],0,gt_h-1e-4)
    xint, yint = junctions[:,0].astype(int), junctions[:,1].astype(int)#向下取整
    off_x = junctions[:,0] - xint#0~1
    off_y = junctions[:,1] - yint
    if gaussian:
        size = round(sigma * 3)
        if size % 2 == 0:
            size += 1
        half_size = size // 2
        for x, y in zip(xint, yint):
            for i in range(max(0, x - half_size), min(gt_w, x + half_size + 1)):
                for j in range(max(0, y - half_size), min(gt_h, y + half_size + 1)):
                    dist_sqare = (i - x) ** 2 + (j - y) ** 2
                    value = np.exp(-0.5 * dist_sqare / sigma ** 2)
                    vmap[j, i] = max(vmap[j, i], value)
    else:
        vmap[yint, xint] = 1
    voff[0, yint, xint] = off_x
    voff[1, yint, xint] = off_y
    return vmap, voff
def get_mask(junctions,seg_h,seg_w):
    #junctions: np.array(n,2) x,y (w,h) (seg_h,seg_w)范围内
    junctions=[junctions.reshape(-1).tolist()]#[[x1,y1],[x2,y2],...] -> [[x1,y1,x2,y2,...]]
    #junctions coco标注'segmentation'字段
    #由多边形顶点列表获取gt_mask, 大小为seg_h,seg_w；
    #获取分割图gt_mask：
    rle = maskUtils.frPyObjects(junctions, seg_h,seg_w)#h,w
    gt_mask = maskUtils.decode(rle)
    return gt_mask
def generate_edge_map(junctions, gt_h, gt_w,gaussian=True):
    #junctions: np.array(n,2) x,y (w,h) (gt_h,gt_w)范围内
    junctions = junctions.astype(np.int32)
    edge_map = np.zeros((gt_h, gt_w), dtype=np.uint8)
    edge_map = cv2.polylines(edge_map, [junctions], isClosed=True, color=1, thickness=1)
    if gaussian:
        edge_map=edge_map.astype(np.float32)
        sigma=1
        size = int(sigma*2)+1
        gaussian_map = cv2.GaussianBlur(edge_map, (size, size), sigma)
        edge_map = gaussian_map / np.max(gaussian_map)
    return edge_map
def clip_polygon_by_bbox(polygon_points: np.ndarray, bbox: tuple):
    """裁剪多边形与边界框的交集部分。    
    Args:
        polygon_points (np.ndarray): 多边形顶点的坐标，形状为(n,2)。
        bbox (tuple): 边界框的四个顶点坐标，格式为(minx, miny, maxx, maxy)。
    Returns:
        np.ndarray: 裁剪后的多边形顶点的坐标。(n,2)
    """
    # 创建Polygon和Box对象
    polygon = Polygon(polygon_points)
    bbox_polygon = box(*bbox)
    # 计算交集
    intersection_polygon = polygon.intersection(bbox_polygon)
    # 如果交集为多边形，则返回其顶点坐标
    if intersection_polygon.is_empty:
        return np.array([])
    else:
        # shapely.coords.CoordinateSequence
        if isinstance(intersection_polygon,(GeometryCollection,MultiPolygon)):
            for geometry in intersection_polygon.geoms:
                if isinstance(geometry, Polygon):
                    # 如果是多边形类型，则执行相应操作
                    polygon = geometry
                    return np.array(polygon.exterior.coords)
        elif isinstance(intersection_polygon,(MultiLineString,LineString)):
            return np.array([])
        else:#Polygon
            return np.array(intersection_polygon.exterior.coords)
        #todo AttributeError: 'Point' object has no attribute 'exterior'
def point_in_bbox(point, bbox):
    """判断点是否在边界框内。
    Args:
        point 点的坐标[x,y]
        bbox (tuple): 边界框的四个顶点坐标，格式为(minx, miny, maxx, maxy)。
    Returns:
        bool: 点是否在边界框内。
    """
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
def point_in_box(x, y, w0, h0, crop_size=512):
    """
    获取在图像长宽范围内，可以包含给定点的裁剪框位置
    Parameters:
    x, y: 点的坐标
    w0, h0: 图像的原始尺寸
    crop_size: 需要的裁剪框的大小，默认512
    Returns:
    x1, y1 裁剪框的左上角坐标
    """
    half_crop_size = crop_size // 2
    off_x=randint(0,20)
    off_x-=10
    off_y=randint(0,20)
    off_y-=10
    # 确定裁剪框中心应该在何处以保证(x, y)在裁剪框内部 默认裁剪框中心为x,y加上(-10,10)之间的随机偏移量
    center_x = min(max(x+off_x, half_crop_size), w0 - half_crop_size)
    center_y = min(max(y+off_y, half_crop_size), h0 - half_crop_size)
    x1 = center_x - half_crop_size
    y1 = center_y - half_crop_size
    return x1, y1
