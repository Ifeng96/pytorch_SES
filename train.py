import Model
import Data
import Loss
import Frame
import log_utils

import torch

import os
import cv2
import time
import numpy as np
from tqdm import tqdm

cfg = {}
cfg.root_path = '/media/hlf/51aa617b-dbbc-4ad1-af81-45cf8dfce172/hlf/data/hyh/allgt/img_gt/out_train_patch1'
cfg.batch_size = 8
cfg.epochs = 100
# log
log_path = './log'
cfg.time_ymdh = time.strftime('%y_%m_%d_%H')
cfg.trainout_path = os.path.join(log_path, cfg.time_ymdh, 'train')
cfg.valout_path= os.path.join(log_path, cfg.time_ymdh, 'val')
if not os.path.exists(log_path): os.makedirs(log_path)
if not os.path.exists(cfg.trainout_path): os.makedirs(cfg.trainout_path)
if not os.path.exists(cfg.valout_path): os.makedirs(cfg.valout_path)
logger = log_utils.init_logger(os.path.join(log_path, cfg.time_ymdh, 'log'))
logger.info(cfg)
# model
cfg.model_path = os.path.join(log_path, cfg.time_ymdh, 'model')
if not os.path.exists(cfg.model_path): os.makedirs(cfg.model_path)
# dataset
Dataset = Data.Data(cfg.root_path)
train_loader = torch.utils.data.DataLoader(
    Dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4
)

frame = Frame.Frame(Model.Model, Loss.dice_bce_loss())

best_loss = 1e4
for epoch in tqdm(range(cfg.epochs)):
    loss_epoch = 0
    for step, (img, gt) in enumerate(iter(train_loader)):
        frame.set_input(img, gt)
        loss, pred = frame.optimize()
        loss_epoch += loss
        if step % 100 == 0:
            cv2.imwrite(os.path.join(cfg.trainout_path, 'epoch{}step{}.tif').format(epoch, step), np.transpose(pred[0], (1,2,0)))
            logger.info('[epoch {}] step {}: loss: {}'.format(epoch+1, step, loss))
    if loss_epoch < best_loss:
        best_loss = loss_epoch
        frame.save(os.path.join(cfg.model_path, 'epoch{}loss{}.pth'.format(epoch, best_loss)))
