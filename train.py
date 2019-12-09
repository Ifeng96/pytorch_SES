import Model
import Data
import Loss
import Frame
import log_utils
from Config import cfg

import torch

import os
import cv2
import numpy as np
from tqdm import tqdm

# log
logger = log_utils.init_logger(cfg['logger_path'])
logger.info(cfg)
# dataset
Dataset = Data.Data(cfg['root_path']+'/train', cfg['loader'])
train_loader = torch.utils.data.DataLoader(
    Dataset,
    batch_size=cfg['batch_size'],
    shuffle=True,
    num_workers=4
)
Dataset_val = Data.Data(cfg['root_path']+'/valid', cfg['loader'])
val_loader = torch.utils.data.DataLoader(
    Dataset_val,
    batch_size=cfg['batch_size'],
    shuffle=True,
    num_workers=4
)

frame = Frame.Frame(Model.Model, Loss.dice_bce_loss())

best_loss = 1e4
for epoch in tqdm(range(cfg['epochs'])):
    loss_epoch = 0
    frame.train()
    for step, (img, gt, _) in enumerate(iter(train_loader)):
        frame.set_input(img, gt)
        loss, pred = frame.optimize()
        loss_epoch += loss
        if step % 100 == 0:
            cv2.imwrite(os.path.join(cfg['trainout_path'], 'epoch{}step{}.tif').format(epoch, step), np.uint8(255*np.transpose(pred[0], (1,2,0))))
            cv2.imwrite(os.path.join(cfg['trainout_path'], 'epoch{}step{}gt.tif').format(epoch, step),
                        np.uint8(255 * np.transpose(frame.gt.cpu().data.numpy()[0], (1, 2, 0))))
            logger.info('train[epoch {}] step {}: loss: {}'.format(epoch+1, step, loss))
    with torch.no_grad():
        frame.eval()
        loss_val = []
        for step, (img, gt, _) in enumerate(iter(val_loader)):
            # torch.no_grad() to avoid cuda out of memory
            frame.set_input(img, gt)
            loss,pred_val = frame.val_op()
            loss_val.append(loss)
            if step % 100 == 0:
                cv2.imwrite(os.path.join(cfg['valout_path'], 'epoch{}step{}.tif').format(epoch, step),
                            np.uint8(255 * np.transpose(pred_val[0], (1, 2, 0))))
                cv2.imwrite(os.path.join(cfg['valout_path'], 'epoch{}step{}gt.tif').format(epoch, step),
                            np.uint8(255 * np.transpose(frame.gt.cpu().data.numpy()[0], (1, 2, 0))))
                logger.info('val[epoch {}] step {}: loss: {}'.format(epoch + 1, step, np.mean(np.array(loss))))
    if loss_epoch < best_loss:
        best_loss = loss_epoch
        frame.save(os.path.join(cfg['model_path'], 'epoch{}loss{}.pth'.format(epoch, best_loss)))
