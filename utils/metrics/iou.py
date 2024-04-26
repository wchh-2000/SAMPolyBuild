from shapely.geometry import Polygon
import cv2
import numpy as np
def IoU(preds, labels,eps=1e-5):
    """
    计算 IoU。
    preds: 预测的分割 masks，形状为 [b, 3/1, h, w] logits值
    labels: 真实的分割 masks，形状为 [b, 1, h, w]
    3/1个预测的mask分别和1个真实的mask计算iou
    返回形状为 [b, 3/1] 的 IoU 矩阵。
    """
    preds = preds > 0 #preds为sigmoid前，等价于sigmoid(preds) > 0.5
    # labels = labels > 0
    if preds.shape[1]>1:
        # 扩展 labels 以匹配 preds 的形状 [b, 3, h, w]
        labels = labels.expand_as(preds)

    # 计算交集和并集
    intersection = (preds & labels).float().sum(dim=(2, 3))
    union = (preds | labels).float().sum(dim=(2, 3))

    # 计算 IoU
    iou = (intersection + eps)/ (union + eps)# eps避免除以零
    return iou

#直接多边形iou
def iou_poly(polygon1, polygon2):
    """
    Calculate the IoU between two polygons.
    
    Parameters:
    - polygon1, polygon2: ndarray(n, 2) coordinates representing the vertices of the polygons.

    Returns:
    - IoU value (float)
    """
    polygon1 = Polygon(polygon1)
    if type(polygon2)!=Polygon:
        polygon2 = Polygon(polygon2)
    polygon1 = polygon1.buffer(0)#修复自相交
    polygon2 = polygon2.buffer(0)
    # Calculate intersection and union areas
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.area + polygon2.area - intersection_area    
    # union_area = polygon1.union(polygon2).area
    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0    
    return iou

# from https://github.com/liweijia/polycity polygon-rnnpp/Evaluation/metrics.py:
def iou_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou
def draw_poly(mask, poly):
    """
    NOTE: Numpy function
    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)
    cv2.fillPoly(mask, [poly.astype(int)], 255)
    return mask
def iou_from_poly(pred, gt, width=None, height=None):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: (list of) np arrays of predicted polygons 
    gt: (list of) np arrays of gt polygons
    width, height: grid_size that the polygons are in
    每个多边形点坐标是一个ndarray,可以多个多边形拼到一个mask上，一起计算iou
    """
    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]
    if width is None or height is None:
        #统计所有多边形的最大边界：
        max_x = max([p[:, 0].max() for p in pred + gt])
        max_y = max([p[:, 1].max() for p in pred + gt])
        width = int(max_x + 1)
        height = int(max_y + 1)
    masks = np.zeros((2, height, width), dtype=np.uint8)    

    for p in pred:
        masks[0] = draw_poly(masks[0], p)
    for g in gt:
        masks[1] = draw_poly(masks[1], g)

    return iou_from_mask(masks[0], masks[1])#, masks

if __name__ == "__main__":
# Example
    poly1 = np.array([(0, 0), (0, 2), (2, 2), (2, 0)])
    poly2 = np.array([(1, 1), (1, 3), (3, 3), (3, 1)])
    print(iou_poly(poly1, poly2))  # Expected: 0.142857
    print(iou_from_poly(poly1, poly2,5,5))#先转mask再计算iou