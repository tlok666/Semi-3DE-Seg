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
#from visualize import Visualizer

class SyntheticDataSet(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 5, Is_Flip = True):
        #self.grid = t.from_numpy((generate_grid([config.image_size, config.image_size, config.image_size]) - config.image_size//2)/32.0).float()
        self.curFolderNum = 0 
        self.folders = ['laddist', 'ladprox', 'lbbbsmall', 'lcx']  #-- 'normal', 'rca', 'sync', 'lbbb'
        self.data_dir = './dataset/Synthetic'
        self.image_size = crop_size
        self.gt_data = {}
        self.files   = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True
        for name in self.folders:
            self.gt_dir = osp.join(self.data_dir, name + '_dense_disps_gt.mat')
            self.gt_data[name] = sio.loadmat(self.gt_dir)['BX_prop']
            #imgs = os.listdir(osp.join(data_dir, name))
            for filename in range(self.gt_data[name].shape[3]):
                img_file = osp.join(self.data_dir, name + '_image_rsp.mat')
                label_file = osp.join(self.data_dir, name + '_dense_disps_gt.mat')
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": filename
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        image = sio.loadmat(datafiles["img"])['bMode_rsp'][:,:,:,datafiles["name"] + 1]
        label = sio.loadmat(datafiles["label"])['BX_prop'][:,:,:,datafiles["name"]]
        label = (((label[:,:,:])!=0).astype(float))*((image[:,:,:]!=0).astype(float))

        
        xShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        yShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        zShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        if self.Is_Flip:
            xflip = randint(0,1)
            yflip = randint(0,1)
            zflip = randint(0,1)
        else:
            xflip = 0
            yflip = 0
            zflip = 0
        grid = t.from_numpy((generate_grid(self.image_size, xShiftRandom, yShiftRandom, zShiftRandom, xflip, yflip, zflip) - max(self.image_size)//2)/float(max(self.image_size)//2)).float()
        image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        label = t.nn.functional.grid_sample(t.from_numpy(label).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        return image, label.squeeze(1), image.shape, str(datafiles["name"])
    
        
class SyntheticNoLabelDataSet(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 5, Is_Flip = True):
        #self.grid = t.from_numpy((generate_grid([config.image_size, config.image_size, config.image_size]) - config.image_size//2)/32.0).float()
        self.curFolderNum = 0 
        self.folders = ['rca', 'sync', 'lbbb']  #-- 'normal', 'rca', 'sync', 'lbbb'
        self.data_dir = './dataset/Synthetic'
        self.image_size = crop_size
        self.gt_data = {}
        self.files   = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True
        for name in self.folders:
            self.gt_dir = osp.join(self.data_dir, name + '_dense_disps_gt.mat')
            self.gt_data[name] = sio.loadmat(self.gt_dir)['BX_prop']
            #imgs = os.listdir(osp.join(data_dir, name))
            for filename in range(self.gt_data[name].shape[3]):
                img_file = osp.join(self.data_dir, name + '_image_rsp.mat')
                label_file = osp.join(self.data_dir, name + '_dense_disps_gt.mat')
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": filename + 1
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        image = sio.loadmat(datafiles["img"])['bMode_rsp'][:,:,:,datafiles["name"]]
        label = sio.loadmat(datafiles["label"])['BX_prop'][:,:,:,datafiles["name"]-1]
        label = (((label[:,:,:])!=0).astype(float))*((image[:,:,:]!=0).astype(float))

        
        xShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        yShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        zShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        if self.Is_Flip:
            xflip = randint(0,1)
            yflip = randint(0,1)
            zflip = randint(0,1)
        else:
            xflip = 0
            yflip = 0
            zflip = 0
        grid = t.from_numpy((generate_grid(self.image_size, xShiftRandom, yShiftRandom, zShiftRandom, xflip, yflip, zflip) - max(self.image_size)//2)/float(max(self.image_size)//2)).float()
        image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        label = t.nn.functional.grid_sample(t.from_numpy(label).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        return image, label.squeeze(1), str(datafiles["name"])
    
class SyntheticTestSet(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 5, Is_Flip = True):
        #self.grid = t.from_numpy((generate_grid([config.image_size, config.image_size, config.image_size]) - config.image_size//2)/32.0).float()
        self.curFolderNum = 0 
        self.folders = ['rca', 'sync', 'lbbb']  #-- 'normal', 'rca', 'sync', 'lbbb'
        self.data_dir = './dataset/Synthetic'
        self.image_size = crop_size
        self.gt_data = {}
        self.files   = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = Is_Flip
        for name in self.folders:
            self.gt_dir = osp.join(self.data_dir, name + '_dense_disps_gt.mat')
            self.gt_data[name] = sio.loadmat(self.gt_dir)['BX_prop']
            #imgs = os.listdir(osp.join(data_dir, name))
            for filename in range(self.gt_data[name].shape[3]):
                img_file = osp.join(self.data_dir, name + '_image_rsp.mat')
                label_file = osp.join(self.data_dir, name + '_dense_disps_gt.mat')
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": filename + 1
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        image = sio.loadmat(datafiles["img"])['bMode_rsp'][:,:,:,datafiles["name"]]
        label = sio.loadmat(datafiles["label"])['BX_prop'][:,:,:,datafiles["name"]-1]
        label = (((label[:,:,:])!=0).astype(float))*((image[:,:,:]!=0).astype(float))

        
        xShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        yShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        zShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        if self.Is_Flip:
            xflip = randint(0,1)
            yflip = randint(0,1)
            zflip = randint(0,1)
        else:
            xflip = 0
            yflip = 0
            zflip = 0
        grid = t.from_numpy((generate_grid(self.image_size, xShiftRandom, yShiftRandom, zShiftRandom, xflip, yflip, zflip) - max(self.image_size)//2)/float(max(self.image_size)//2)).float()
        image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        label = t.nn.functional.grid_sample(t.from_numpy(label).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        return image, label.squeeze(1), image.shape, str(datafiles["name"])    
    
    
class SyntheticDataSet3D(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 5, Is_Flip = True):
        #self.grid = t.from_numpy((generate_grid([config.image_size, config.image_size, config.image_size]) - config.image_size//2)/32.0).float()
        self.curFolderNum = 0 
        self.folders = ['oneshot']  #--'laddist', 'ladprox', 'lbbbsmall', 'lcx' #-- 'normal', 'rca', 'sync', 'lbbb'
        self.data_dir = './dataset/Synthetic'
        self.image_size = crop_size
        self.dir = ''
        self.files = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True

        for name in self.folders:
            data_dir_ = osp.join(self.data_dir, name)
            imgs = os.listdir(osp.join(data_dir_, 'raw'))
            for filename in imgs:
                img_file = osp.join(osp.join(data_dir_, 'raw'), filename)
                label_file = osp.join(osp.join(data_dir_, 'gt'), filename)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": filename
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        image = sio.loadmat(datafiles["img"])['bMode']
        label = sio.loadmat(datafiles["label"])['bMode']
        label = (((label[:,:,:])!=0).astype(float))*((image[:,:,:]!=0).astype(float))

        
        xShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        yShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        zShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        if self.Is_Flip:
            xflip = randint(0,1)
            yflip = randint(0,1)
            zflip = randint(0,1)
        else:
            xflip = 0
            yflip = 0
            zflip = 0
        grid = t.from_numpy((generate_grid(self.image_size, xShiftRandom, yShiftRandom, zShiftRandom, xflip, yflip, zflip) - max(self.image_size)//2)/float(max(self.image_size)//2)).float()
        image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        label = t.nn.functional.grid_sample(t.from_numpy(label).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        return image, label.squeeze(1), image.shape, str(datafiles["name"])
    

class SyntheticDataSet3D_Gt(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 5, Is_Flip = True):
        #self.grid = t.from_numpy((generate_grid([config.image_size, config.image_size, config.image_size]) - config.image_size//2)/32.0).float()
        self.curFolderNum = 0 
        self.folders = ['laddist', 'ladprox', 'lbbbsmall', 'lcx', 'rca', 'sync', 'lbbb']  #--'laddist', 'ladprox', 'lbbbsmall', 'lcx' #-- 'normal', 'rca', 'sync', 'lbbb'
        self.data_dir = './dataset/Synthetic'
        self.image_size = crop_size
        self.dir = ''
        self.files = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True

        for name in self.folders:
            data_dir_ = osp.join(self.data_dir, name)
            imgs = os.listdir(osp.join(data_dir_, 'raw'))
            for filename in imgs:
                img_file = osp.join(osp.join(data_dir_, 'raw'), filename)
                label_file = osp.join(osp.join(data_dir_, 'gt'), filename)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": filename
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        image = sio.loadmat(datafiles["img"])['bMode']
        label = sio.loadmat(datafiles["label"])['bMode']
        label = (((label[:,:,:])!=0).astype(float))*((image[:,:,:]!=0).astype(float))

        
        xShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        yShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        zShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        if self.Is_Flip:
            xflip = randint(0,1)
            yflip = randint(0,1)
            zflip = randint(0,1)
        else:
            xflip = 0
            yflip = 0
            zflip = 0
        grid = t.from_numpy((generate_grid(self.image_size, xShiftRandom, yShiftRandom, zShiftRandom, xflip, yflip, zflip) - max(self.image_size)//2)/float(max(self.image_size)//2)).float()
        image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        label = t.nn.functional.grid_sample(t.from_numpy(label).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        return image, label.squeeze(1), image.shape, str(datafiles["name"])

       
class SyntheticNoLabelDataSet3D(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 5, Is_Flip = True):
        #self.grid = t.from_numpy((generate_grid([config.image_size, config.image_size, config.image_size]) - config.image_size//2)/32.0).float()
        self.curFolderNum = 0 
        self.folders = ['laddist', 'ladprox', 'lbbbsmall', 'lcx']  #-- 'normal', 'rca', 'sync', 'lbbb'
        self.data_dir = './dataset/Synthetic'
        self.image_size = crop_size
        self.dir = ''
        self.files = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True

        for name in self.folders:
            data_dir_ = osp.join(self.data_dir, name)
            imgs = os.listdir(osp.join(data_dir_, 'raw'))
            for filename in imgs:
                img_file = osp.join(osp.join(data_dir_, 'raw'), filename)
                label_file = osp.join(osp.join(data_dir_, 'gt'), filename)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": filename
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        image = sio.loadmat(datafiles["img"])['bMode']
        label = sio.loadmat(datafiles["label"])['bMode']
        label = (((label[:,:,:])!=0).astype(float))*((image[:,:,:]!=0).astype(float))

        
        xShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        yShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        zShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        if self.Is_Flip:
            xflip = randint(0,1)
            yflip = randint(0,1)
            zflip = randint(0,1)
        else:
            xflip = 0
            yflip = 0
            zflip = 0
        grid = t.from_numpy((generate_grid(self.image_size, xShiftRandom, yShiftRandom, zShiftRandom, xflip, yflip, zflip) - max(self.image_size)//2)/float(max(self.image_size)//2)).float()
        image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        label = t.nn.functional.grid_sample(t.from_numpy(label).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        return image, label.squeeze(1), str(datafiles["name"])
    
class SyntheticTestSet3D(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 0, Is_Flip = False):
        #self.grid = t.from_numpy((generate_grid([config.image_size, config.image_size, config.image_size]) - config.image_size//2)/32.0).float()
        self.curFolderNum = 0 
        self.folders = ['rca', 'sync', 'lbbb']  #-- 'normal', 'rca', 'sync', 'lbbb'
        self.data_dir = './dataset/Synthetic'
        self.image_size = crop_size
        self.dir = ''
        self.files = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True

        for name in self.folders:
            data_dir_ = osp.join(self.data_dir, name)
            imgs = os.listdir(osp.join(data_dir_, 'raw'))
            for filename in imgs:
                img_file = osp.join(osp.join(data_dir_, 'raw'), filename)
                label_file = osp.join(osp.join(data_dir_, 'gt'), filename)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": filename
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        image = sio.loadmat(datafiles["img"])['bMode']
        label = sio.loadmat(datafiles["label"])['bMode']
        label = (((label[:,:,:])!=0).astype(float))*((image[:,:,:]!=0).astype(float))

        
        xShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        yShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        zShiftRandom = randint(-self.Max_Shift, self.Max_Shift)
        if self.Is_Flip:
            xflip = randint(0,1)
            yflip = randint(0,1)
            zflip = randint(0,1)
        else:
            xflip = 0
            yflip = 0
            zflip = 0
        grid = t.from_numpy((generate_grid(self.image_size, xShiftRandom, yShiftRandom, zShiftRandom, xflip, yflip, zflip) - max(self.image_size)//2)/float(max(self.image_size)//2)).float()
        image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        label = t.nn.functional.grid_sample(t.from_numpy(label).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        return image, label.squeeze(1), image.shape, str(datafiles["name"])
    
    
def generate_grid(imgshape, xShift=0, yShift=0, zShift=0, xflip=0, yflip=0, zflip=0, batch_size=1):
    x = np.linspace(0+xShift, imgshape[0]-1+xShift, imgshape[0])
    if xflip:
        x = imgshape[0] - x
    y = np.linspace(0+yShift, imgshape[1]-1+yShift, imgshape[1])
    if yflip:
        y = imgshape[0] - y
    z = np.linspace(0+zShift, imgshape[2]-1+zShift, imgshape[2])
    if zflip:
        z = imgshape[0] - z
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    grid_out = grid
    if batch_size > 1:
        grid_out = np.zeros([batch_size, grid.shape[0], grid.shape[1], grid.shape[2], 3])
        for iCnt in range(batch_size):
            grid_out[iCnt,:,:,:] = grid
    return grid_out
    

if __name__ == '__main__':
    #vis = Visualizer('GAN_Segmentation_DataLoader')
    
    #---CardiacDataSet
    files = []
    image_size = (128, 128, 128)
    curFolderNum = 0 
    folders = ['laddist', 'ladprox', 'lbbb', 'lbbbsmall', 'lcx', 'rca', 'sync']
    data_dir = '/home/dragon/Downloads/AdvSemiSeg/dataset/Synthetic'
    for name in folders:
        data_dir_ = osp.join(data_dir, name)
        imgs = os.listdir(osp.join(data_dir_, 'raw'))
        for filename in imgs:
            img_file = osp.join(osp.join(data_dir_, 'raw'), filename)
            label_file = osp.join(osp.join(data_dir_, 'gt'), filename)
            files.append({
                "img": img_file,
                "label": label_file,
                "name": filename
                })
    index = 1
    datafiles = files[index]
    image = sio.loadmat(datafiles["img"])['bMode']
    label = sio.loadmat(datafiles["label"])['bMode']
    label = (((label[:,:,:])!=0).astype(float))*((image[:,:,:]!=0).astype(float))
    #vis.heatmap((image[:,70,:]), win='Input')
    #vis.heatmap((label[:,70,:]), win='Label')
    
    Is_Flip = True
    Max_Shift = 5
    xShiftRandom = randint(-Max_Shift, Max_Shift)
    yShiftRandom = randint(-Max_Shift, Max_Shift)
    zShiftRandom = randint(-Max_Shift, Max_Shift)
    if Is_Flip:
        xflip = randint(0,1)
        yflip = randint(0,1)
        zflip = randint(0,1)
    else:
        xflip = 0
        yflip = 0
        zflip = 0
    grid = t.from_numpy((generate_grid(image_size, xShiftRandom, yShiftRandom, zShiftRandom, xflip, yflip, zflip) - max(image_size)//2)/float(max(image_size)//2)).float()
    image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
    label = t.nn.functional.grid_sample(t.from_numpy(label).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
    vis.heatmap((image[0,0,64,:,:]), win='Input')
    vis.heatmap((label[0,0,64,:,:]), win='Label')
        
    
    #---CardiacNoLabelDataSet
    '''
    image_size = (128, 128, 128)
    name  = 'No_label'
    data_dir = '/home/dragon/Downloads/trails/AdvSemiSeg/dataset/Masks_for_Animal_data_Zhao'
    imgs_data  = {}
    files    = []
    Max_Shift = 50
    Is_Flip = True
    
    imgs = os.listdir(osp.join(data_dir, name))
    for filename in imgs:
        imgs_data[filename] = sio.loadmat(osp.join(osp.join(data_dir, name), filename))['bMode']
        for iCnt in range(imgs_data[filename].shape[3]):
            label_file = iCnt
            files.append({
                "img": None,
                "label": label_file,
                "name": filename
                })
    
    index = 5
    datafiles = files[index]  
    image = imgs_data[datafiles["name"]][:,:,:,datafiles["label"]]
    label = None
    
    xShiftRandom = randint(-Max_Shift, Max_Shift)
    yShiftRandom = randint(-Max_Shift, Max_Shift)
    zShiftRandom = randint(-Max_Shift, Max_Shift)
    if Is_Flip:
        xflip = randint(0,1)
        yflip = randint(0,1)
        zflip = randint(0,1)
    else:
        xflip = 0
        yflip = 0
        zflip = 0
    grid = t.from_numpy((generate_grid(image_size, xShiftRandom, yShiftRandom, zShiftRandom, xflip, yflip, zflip) - max(image_size)//2)/float(max(image_size)//2)).float()
    image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear')
    label = None
    image = image.squeeze(0).numpy()
    label = None
    vis.images((image[0,32,:,:]), win='Input')
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    