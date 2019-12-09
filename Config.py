# -*- coding:utf-8 -*-

# @Filename: Config
# @Project : pytorch_SES
# @date    : 2019-12-09 21:33
# @Author  : ifeng
import time
import sys
import os
print(sys.argv[0])

cfg = {}
# cfg['root_path'] = '/media/hlf/51aa617b-dbbc-4ad1-af81-45cf8dfce172/hlf/data/hyh/allgt/img_gt/out_train_patch1'
cfg['root_path'] = '/media/hlf/51aa617b-dbbc-4ad1-af81-45cf8dfce172/hlf/data/mass_road'
cfg['batch_size'] = 8
cfg['epochs'] = 100
cfg['loader'] = 'mass_loader'
# log
cfg['log_path'] = './log'
cfg['time_ymdh'] = time.strftime('%y_%m_%d_%H')
cfg['logger_path'] = os.path.join(cfg['log_path'], cfg['time_ymdh'], 'log')
cfg['trainout_path'] = os.path.join(cfg['log_path'], cfg['time_ymdh'], 'train')
cfg['valout_path']= os.path.join(cfg['log_path'], cfg['time_ymdh'], 'val')
# model
cfg['model_path'] = os.path.join(cfg['log_path'], cfg['time_ymdh'], 'model')
# TRAIN
if 'train.py' in sys.argv[0]:
    if not os.path.exists(cfg['trainout_path']): os.makedirs(cfg['trainout_path'])
    if not os.path.exists(cfg['valout_path']): os.makedirs(cfg['valout_path'])
    if not os.path.exists(cfg['model_path']): os.makedirs(cfg['model_path'])

# test
cfg['with_gt'] = True
cfg['test_time'] = '19_12_09_16'
cfg['test_model_file'] = 'epoch99loss26.92005242407322.pth'
cfg['testout_path']= os.path.join(cfg['log_path'], cfg['test_time'], 'test')
cfg['test_model_path'] = os.path.join(cfg['log_path'], cfg['test_time'], 'model')
if 'test.py' in sys.argv[0]:
    if not os.path.exists(cfg['testout_path']): os.makedirs(cfg['testout_path'])
