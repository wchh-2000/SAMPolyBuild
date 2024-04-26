import numpy as np
from tqdm import tqdm

# modified from https://github.com/liweijia/polycity polygon-rnnpp/Evaluation/metrics.py
def precision_recall_from_vertex_set(pred_vertices, gt_vertices):
    #pred_vertices,gt_vertices: ndarray(n,2)
    threshold = 3 #pixels 距离小于阈值的预测点被认为是正确的
    count = 0
    for point in gt_vertices:
        if not pred_vertices.size>0:
            continue
        diff = pred_vertices - point if pred_vertices.size > 0  else 0
        diff_dist = np.square(diff).sum(1)
        diff_count = diff_dist[diff_dist < threshold * threshold]
        if len(diff_count) > 0:
            count = count + 1
    recall = count/len(gt_vertices) if len(gt_vertices) else 0
    count =0
    for point in pred_vertices:
        if not gt_vertices.size>0:
            continue
        diff = gt_vertices - point
        diff_dist = np.square(diff).sum(1)
        diff_count = diff_dist[diff_dist < threshold * threshold]
        if len(diff_count) > 0:
            count = count + 1
    precision = count / len(pred_vertices) if len(pred_vertices) else 0
    return precision, recall


if __name__ == "__main__":
    pred_vertices = np.array([[1, 2], [2, 3], [3, 4], [10,10], [6, 6]])
    gt_vertices = np.array([[1, 2], [2, 3.5], [3, 4], [6, 7]])

    precision, recall = precision_recall_from_vertex_set(pred_vertices, gt_vertices)
    print("Precision:", precision)
    print("Recall:", recall)