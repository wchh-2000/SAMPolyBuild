import torch
import torch.nn as nn
import torch.nn.functional as F
#sigmoid_l1_loss from hisup:
def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset 
    loss = torch.abs(logp-targets)
    if mask is not None:
        t = (mask >0.7).float()#(mask >0.7)高斯核为3的三种归一化值：0.36787948 0.6065307  1. 自动模式线性插值后>0.7的认为是原始mask为1的像素
        w = t.mean(3, True).mean(2,True)#h,w维度求平均，得到每张图的mask为1的像素数
        w[w==0] = 1 #shape:[b, 1, 1, 1] 各图中n/(w*h) n为mask为1或2的像素数
        # 某图只有t全为0（mask都为0）时w才为0，此时loss为0 置1防止除0
        loss = loss*(t/w)#mask为0的位置loss为0，不计入损失
    return loss.mean()#mask为1处的loss求和除以mask为1的像素数n
class BCEDiceLoss(nn.Module):
    def __init__(self,pos_weight=5.):
        super().__init__()
        self.pos_weight=pos_weight
    def __call__(self, preds, targets):
        bce=binary_crossentropy(torch.sigmoid(preds), targets,pos_weight=self.pos_weight)
        dice=sigmoid_dice_loss(preds, targets)
        return 2 * bce + dice
    
def binary_crossentropy(pr, gt, eps=1e-7, pos_weight=1., neg_weight=1.):
    pr = torch.clamp(pr, eps, 1. - eps)
    gt = torch.clamp(gt, eps, 1. - eps)
    loss = - pos_weight * gt * torch.log(pr) -  neg_weight * (1 - gt) * torch.log(1 - pr)
    return torch.mean(loss)

#modified from detrex losses:
def sigmoid_dice_loss(
    preds,
    targets,
    eps: float = 1e-5,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        preds (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor):
            A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-4.

    Return:
        torch.Tensor: The computed dice loss.
    """
    preds=torch.sigmoid(preds)
    preds = preds.flatten(1)#第二个维度及以后的所有元素都被展平为一个维度
    targets = targets.flatten(1).float()
    numerator = 2 * torch.sum(preds * targets, 1) + eps
    denominator = torch.sum(preds, 1) + torch.sum(targets, 1) + eps
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()
def sigmoid_focal_loss(
    preds,
    targets,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        preds (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss with the reduction option applied.
    """
    preds = preds.float()
    targets = targets.float()
    p = torch.sigmoid(preds)
    ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()
def focal_dice_loss(preds,targets,weight=(10,0.5)):#focal_dice_loss in sam 20:1
    return weight[0]*sigmoid_focal_loss(preds,targets)+weight[1]*sigmoid_dice_loss(preds,targets)



if __name__=='__main__':
    # Create sample prediction and ground truth tensors
    batch_size = 2
    width = 32
    height = 32

    predictions = torch.randn(batch_size, 1, width, height)
    ground_truth = torch.randint(0, 2, (batch_size, 1, width, height)).long()
    gtf=ground_truth.float()
    # Compute the losses
    focal_loss_value = sigmoid_focal_loss(predictions, ground_truth)
    dice_loss_value = sigmoid_dice_loss(predictions, ground_truth)

    # Print the computed losses
    print("Focal Loss:", focal_loss_value.item())
    print("Dice Loss:", dice_loss_value.item(),sigmoid_dice_loss(predictions, gtf).item())
    print("Focal+Dice Loss:", focal_dice_loss(predictions, ground_truth).item())
