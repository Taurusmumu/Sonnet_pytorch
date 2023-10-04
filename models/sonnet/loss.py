import numpy as np
import os
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def check_weight_loss(train_dir, dataset):
    '''
        Calculate the weights using for focal_loss_modified()
        Args:
            train_dir: training directory
        Return:
            w: numpy array
    '''
    N = {}
    for file_name in os.listdir(train_dir):
        file_path = os.path.join(train_dir, file_name)
        type_map = scipy.io.loadmat(file_path)['type_map']
        val, cnt = np.unique(type_map, return_counts=True)
        for idx, type_id in enumerate(val):
            N[type_id] = N.get(type_id, 0) + cnt[idx]
    N = sorted(N.items())
    N = [val for key, val in N]
    c = len(N)
    N = np.array(N)
    w = np.power(N[0]/N, 1/3)
    w = w/w.sum() * c
    print(f"{dataset} nt weight: {np.around(w, 3)}")

    N = {}
    for file_name in os.listdir(train_dir):
        file_path = os.path.join(train_dir, file_name)
        fore_map = scipy.io.loadmat(file_path)['type_map']
        fore_map[(fore_map > 0)] = 1
        val, cnt = np.unique(fore_map, return_counts=True)
        for idx, type_id in enumerate(val):
            N[type_id] = N.get(type_id, 0) + cnt[idx]
    N = sorted(N.items())
    N = [val for key, val in N]
    c = len(N)
    N = np.array(N)
    w = np.power(N[0] / N, 1 / 3)
    w = w / w.sum() * c
    print(f"{dataset} nf weight: {np.around(w, 3)}")

    return w


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, prefix=None, pred_no=None):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = torch.argmax(predict, dim=-1)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class TypeFocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=None, size_average=True, has_weight=True):
        super(TypeFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.has_weight = has_weight

    def forward(self, input, target, prefix, pred_no=None):
        input = input.view(-1, input.size(-1))  # N,H,W, C => N*H*W, C
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(-1, target.type(torch.int64))
        logpt = logpt.view(-1)
        target = target.view(-1)

        pt = Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        weight = torch.ones(loss.size()).to(target.device)
        if pred_no is not None:
            pred_no_flat = pred_no.view(-1).to(target.device)
            weight[pred_no_flat == 1] = 3
            weight[pred_no_flat == 2] = 2
        for idx in range(len(prefix)):
            weight[target == idx] *= prefix[idx]

        if self.has_weight is True:
            loss *= weight
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ForeFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, has_weight=True):
        super(ForeFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.has_weight = has_weight

    def forward(self, input, target, prefix, pred_no=None):
        input = input.view(-1, input.size(-1))  # N,H,W, C => N*H*W, C
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(-1, target.type(torch.int64))
        logpt = logpt.view(-1)
        target = target.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        weight = torch.ones(loss.size()).to(target.device)
        if pred_no is not None:
            pred_no = pred_no.view(-1).to(target.device)
            weight[pred_no == 1] = 3
            weight[pred_no == 2] = 2
        for idx in range(len(prefix)):
            weight[target == idx] *= prefix[idx]

        if self.has_weight is True:
            loss *= weight
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class OrdinalFocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True, has_weight=True):
        super(OrdinalFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.has_weight = has_weight

    def forward(self, pre_log_prediction, target, prefix=None, prediction=None):
        num_classes = 8
        N, H, W = target.size()[0], target.size()[1], target.size()[2]
        log_prediction = F.log_softmax(pre_log_prediction, dim=-1) # (N, H * W, 8, 2)
        log_prediction = log_prediction.view(N, -1, num_classes, 2)
        target = target.view(N, -1, 1)  # (N, H * W, 1)
        target_with_level = sequence_mask(target, num_classes)  # (N, H * W, 8, 1)
        logpt = log_prediction.gather(-1, target_with_level.type(torch.int64)).to(target.device)  # (N, H * W, 8, 1)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.has_weight is True:
            weight = torch.ones(loss.size()).to(target.device)
            if prediction is not None:
                prediction = prediction.unsqueeze(-1).expand(N, H, W, num_classes).flatten().to(target.device)
                weight[prediction == 1] = 3
                weight[prediction == 2] = 2
            loss *= weight
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def sequence_mask(target, num_classes):
    row_vector = torch.arange(0, num_classes, 1).to(target.device)
    mask = (row_vector < target).type(torch.int64).to(target.device)
    return mask.unsqueeze(-1)


if __name__ == "__main__":
    dataset = "CoNSeP"
    train_label_dir = f"{os.path.dirname(os.path.dirname(os.getcwd()))}/dataset/{dataset}/Train/Labels"
    weight = check_weight_loss(train_label_dir, dataset)



