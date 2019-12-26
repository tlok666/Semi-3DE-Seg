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
    curFolderNum = 0 
    folders = ['labeled', 'No_label']
    data_dir = './dataset/Masks_for_Animal_data_Zhao'
    iCnt = 0
    for name in folders:
        imgs4Dlist = os.listdir(osp.join(data_dir, name))
        for img4D in imgs4Dlist:
            data_dir_data = osp.join(osp.join(data_dir, name), img4D)
            data_data = sio.loadmat(data_dir_data)['bMode']
            save_path = osp.join(osp.join(osp.join(data_dir, name) + '_', img4D[0:len(img4D)-4]))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for iCnt in range(0, data_data.shape[3]):
                img3D = data_data[:,:,:,iCnt]
                sio.savemat(osp.join(save_path, 'data'+str(iCnt)+'.mat'), {'bMode':img3D})
                iCnt = iCnt+1
            
            
        