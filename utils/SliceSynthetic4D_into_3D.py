# -*- coding: utf-8 -*-


import re
import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image
import torch as t
import scipy.io as sio
from random import randint 

if __name__ == '__main__':   
    #---CardiacDataSet
    files = []
    image_size = (128, 128, 128)
    curFolderNum = 0 
    folders = ['lcx', 'rca', 'sync', 'lbbb', 'laddist', 'ladprox', 'lbbbsmall']
    data_dir = './dataset/Synthetic'
    for name in folders:
        imgs4D_gt  = sio.loadmat(osp.join(data_dir, name) + '_dense_disps_gt.mat')['BX_prop']
        imgs4D_raw = sio.loadmat(osp.join(data_dir, name) + '_image_rsp.mat')['bMode_rsp']
        for iCnt in range(1, imgs4D_raw.shape[3]):
            save_path = osp.join(osp.join(data_dir, name))
            img3D_gt  = imgs4D_gt[:,:,:,iCnt-1]
            img3D_raw = imgs4D_raw[:,:,:,iCnt]
            gt_path  = osp.join(save_path, 'gt')
            raw_path = osp.join(save_path, 'raw')
            if not os.path.exists(gt_path):
                os.makedirs(gt_path)
            if not os.path.exists(raw_path):
                os.makedirs(raw_path)
            sio.savemat(osp.join(gt_path, 'data'+str(iCnt)+'.mat'), {'bMode':img3D_gt})
            sio.savemat(osp.join(raw_path, 'data'+str(iCnt)+'.mat'), {'bMode':img3D_raw})
            
            
        
