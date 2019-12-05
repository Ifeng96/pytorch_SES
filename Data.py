from torch.utils import data
from glob import glob
import torch
import cv2
import os

class Data(data.Dataset):
    def __init__(self, root):
        self.loader = default_loader
        self.root = root

    def __getitem__(self, item):
        name = self.ids[item]
        img, gt = self.loader(name, self.root)
        img = torch.Tensor(img)
        gt = torch.Tensor(gt)
        return img, gt

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def default_loader(name):
        img = cv2.imread(name)
        gt = cv2.imread(name.replace('id', 'regt_id').replace('.tif', '.bmp'))
        gt[gt<128] = 0
        gt[gt>=128] = 1
        return img, gt

    @property
    def ids(self):
        return glob(self.root+'/*.tif')

if __name__ == '__main__':
    Temp_data = Data
    Temp_data.default_loader()