from torch.utils import data
from glob import glob
import numpy as np
import torch
import cv2
import os


def default_loader(name):
    img = cv2.imread(name)
    gt = cv2.imread(name.replace('id', 'regt_id').replace('.tif', '.bmp'), 0)
    gt[gt < 128] = 0
    gt[gt >= 128] = 1
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    gt = gt[:,:,np.newaxis]
    gt = np.array(gt, np.float32).transpose(2, 0, 1)
    return img, gt


class Data(data.Dataset):
    def __init__(self, root):
        self.loader = default_loader
        self.root = root

    def __getitem__(self, item):
        name = self.ids[item]
        img, gt = self.loader(name)
        img = torch.Tensor(img)
        gt = torch.Tensor(gt)
        return img, gt

    def __len__(self):
        return len(self.ids)

    @property
    def ids(self):
        return glob(self.root+'/*.tif')

if __name__ == '__main__':
    Temp_data = Data
    Temp_data.default_loader()