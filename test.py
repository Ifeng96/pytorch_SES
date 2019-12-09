# -*- coding:utf-8 -*-

# @Filename: test
# @Project : pytorch_SES
# @date    : 2019-12-09 21:32
# @Author  : ifeng
import Model
import Data
import Loss
import Frame
import log_utils
from Config import cfg

import torch

import os
import cv2
import time
import numpy as np
from tqdm import tqdm

assert cfg['test_time']!='', 'value of cfg[test_time] is non-exist'
assert cfg['test_model_file']!='', 'value of cfg[test_model_file] is non-exist'

# dataset
Dataset_test = Data.Data(cfg['root_path']+'/test', cfg['loader'])
test_loader = torch.utils.data.DataLoader(
    Dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=1
)
frame = Frame.Frame(Model.Model, Loss.dice_bce_loss())
frame.load(os.path.join(cfg['test_model_path'], cfg['test_model_file']))

frame.eval()
for step, (img, gt, name) in enumerate(iter(test_loader)):
    frame.set_input(img, gt)
    _, pred = frame.val_op()
    basename = os.path.basename(name[0])
    cv2.imwrite(os.path.join(cfg['testout_path'], '{}pred.bmp').format(basename),
                np.uint8(255 * np.transpose(pred[0], (1, 2, 0))))
    cv2.imwrite(os.path.join(cfg['testout_path'], '{}gt.bmp').format(basename),
                np.uint8(255 * np.transpose(frame.gt.cpu().data.numpy()[0], (1, 2, 0))))
    # logger.info('train[epoch {}] step {}: loss: {}'.format(epoch + 1, step, loss))

