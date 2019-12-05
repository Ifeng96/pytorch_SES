import torch
from torch import nn

class dice_bce_loss(nn.Module):
    def __init__(self):
        super(dice_bce_loss, self).__init__()
        self.bce_loss = nn.BCELoss

    def dice_loss(self, gt, pred):
        smooth = 1e-5
        i = torch.sum(gt)
        j = torch.sum(pred)
        intersection = torch.sum(gt*pred)
        score = (intersection+smooth)/(i+j-intersection+smooth)
        return score.mean()

    def __call__(self, gt, pred):
        return self.dice_loss(gt, pred) + self.bce_loss(pred, gt)
