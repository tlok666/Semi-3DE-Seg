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

class CardiacDataSet(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 20, Is_Flip = True):
        self.image_size = crop_size
        self.curFolderNum = 0 
        self.folders  = ['DSEC9BrfDOBSTR', 'dsec9Arfla3 baseline'] #--'dsec9Arfla3 baseline'  'DSEC9 baseline'
        self.data_dir = './dataset/Masks_for_Animal_data_Zhao'
        self.gt_data  = {}
        self.files    = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True
        
        for name in self.folders:
            self.gt_dir = osp.join(self.data_dir, name + '.mat')
            self.imgs = os.listdir(osp.join(self.data_dir, name))
            self.gt_data[name] = sio.loadmat(self.gt_dir)['bMode']
            for filename in self.imgs:
                img_file = osp.join(osp.join(self.data_dir, name), filename)
                label_file = int(re.findall(r"\d+d*",filename)[0])-1
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        label = sio.loadmat(datafiles["img"])['myomask_revserse']
        image = self.gt_data[datafiles["name"]][:,:,:,datafiles["label"]]
        
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
        if self.Is_Flip:
            ilist= [2, 3, 4]
            idex1 = randint(0, 2)
            iExlusive = randint(1,2)
            idex2 = (idex1+iExlusive)%len(ilist)
            idex3 = (idex1+(len(ilist)-iExlusive))%len(ilist)
            image = image.permute((0, 1, ilist[idex1], ilist[idex2], ilist[idex3]))
            label = label.permute((0, 1, ilist[idex1], ilist[idex2], ilist[idex3]))
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        label = (label.squeeze(0).numpy()!=0).astype(float)
        return image.copy(), label.copy(), image.shape, datafiles["name"] + '_' + str(datafiles["label"])
    
class CardiacDataSet3D(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 20, Is_Flip = True):
        self.image_size = crop_size
        self.curFolderNum = 0 
        self.folders  = ['DSEC9BrfDOBSTR', 'dsec9Arfla3 baseline'] #--'dsec9Arfla3 baseline'  'DSEC9 baseline'
        self.data_dir = './dataset/Masks_for_Animal_data_Zhao'
        #self.gt_data  = {}
        self.files    = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True
        
        for name in self.folders:
            self.gt_dir = osp.join(self.data_dir, name + '.mat')
            self.imgs = os.listdir(osp.join(self.data_dir, name))
            #self.gt_data[name] = sio.loadmat(self.gt_dir)['bMode']
            for filename in self.imgs:
                img_file = osp.join(osp.join(self.data_dir, name), filename)
                label_file = int(re.findall(r"\d+d*",filename)[0])-1
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        label = sio.loadmat(datafiles["img"])['myomask_revserse']
        path  = osp.join(osp.join(osp.join(self.data_dir, 'labeled_'), datafiles["name"]), 'data'+str(datafiles["label"])+'.mat')
        image = sio.loadmat(path)['bMode']
        
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
        if self.Is_Flip:
            ilist= [2, 3, 4]
            idex1 = randint(0, 2)
            iExlusive = randint(1,2)
            idex2 = (idex1+iExlusive)%len(ilist)
            idex3 = (idex1+(len(ilist)-iExlusive))%len(ilist)
            image = image.permute((0, 1, ilist[idex1], ilist[idex2], ilist[idex3]))
            label = label.permute((0, 1, ilist[idex1], ilist[idex2], ilist[idex3]))
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        label = (label.squeeze(0).numpy()!=0).astype(float)
        return image.copy(), label.copy(), image.shape, datafiles["name"] + '_' + str(datafiles["label"])    
    
class CardiacNoLabelDataSet(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 15, Is_Flip = True):
        self.image_size = crop_size
        self.name  = 'No_label'
        self.data_dir = './dataset/Masks_for_Animal_data_Zhao'
        self.imgs_data  = {}
        self.files    = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True
        
        self.imgs = os.listdir(osp.join(self.data_dir, self.name))
        for filename in self.imgs:
            self.imgs_data[filename] = sio.loadmat(osp.join(osp.join(self.data_dir, self.name), filename))['bMode']
            for iCnt in range(self.imgs_data[filename].shape[3]):
                label_file = iCnt
                self.files.append({
                    "img": None,
                    "label": label_file,
                    "name": filename
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            datafiles = self.files[index]  
        except:
            print("----------------------------")
            print(index)
            print("----------------------------")
                
        image = self.imgs_data[datafiles["name"]][:,:,:,datafiles["label"]]
        
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
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        return image.copy(), image.shape, datafiles["name"] + '_' + str(datafiles["label"])
    
class CardiacTestDataSet(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 0, Is_Flip = False):
        self.image_size = crop_size
        self.curFolderNum = 0 
        self.folders  = ['dsec9Arfla3 baseline']  # 'dsec9Arfla3 baseline' 'DSEC9 baseline'
        self.data_dir = './dataset/Masks_for_Animal_data_Zhao'
        self.gt_data  = {}
        self.files    = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True
        
        for name in self.folders:
            self.gt_dir = osp.join(self.data_dir, name + '.mat')
            self.imgs = os.listdir(osp.join(self.data_dir, name))
            self.gt_data[name] = sio.loadmat(self.gt_dir)['bMode']
            for filename in self.imgs:
                img_file = osp.join(osp.join(self.data_dir, name), filename)
                label_file = int(re.findall(r"\d+d*",filename)[0])-1
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]        
        label = sio.loadmat(datafiles["img"])['myomask_revserse']
        image = self.gt_data[datafiles["name"]][:,:,:,datafiles["label"]]
        
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
        label = (label.squeeze(0).numpy()!=0).astype(float)
        
        #--https://blog.csdn.net/jessican_uestc/article/details/79541582
        #image = image.transpose((0,1,3,2))
        #label = label.transpose((0,1,3,2))
        
        return image.copy(), label.copy(), image.shape, datafiles["name"] + '_' + str(datafiles["label"])


class CardiacNoLabelDataSet_3D(data.Dataset):
    def __init__(self, crop_size=(128, 128, 128), Max_Shift = 15, Is_Flip = True):
        self.image_size = crop_size
        self.name  = 'No_label_'
        self.data_dir = './dataset/Masks_for_Animal_data_Zhao'
        self.files    = []
        self.Max_Shift = Max_Shift
        self.Is_Flip = True
        self.imgs_folder = os.listdir(osp.join(self.data_dir, self.name))
        
        iCnt = 0
        for file_folder_name in self.imgs_folder:
            imgs = os.listdir(osp.join(osp.join(self.data_dir, self.name), file_folder_name))
            for filename in imgs:
                self.files.append({
                    "img": None,
                    "label": iCnt,
                    "name": osp.join(osp.join(osp.join(self.data_dir, self.name), file_folder_name), filename)
                    })
                iCnt = iCnt + 1    
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            datafiles = self.files[index]  
        except:
            print("----------------------------")
            print(index)
            print("----------------------------")
                
        image_dir = datafiles["name"]
        image = sio.loadmat(image_dir)['bMode']
        
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
        image = (image.squeeze(0).numpy() - 127.0)/128.0
        return image.copy(), image.shape, datafiles["name"] + '_' + str(datafiles["label"])
    
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
    #---CardiacnoLabelDataSet3D
    '''
    image_size = (128, 128, 128)
    name  = 'No_label_'
    data_dir = '/home/dragon/Downloads/AdvSemiSeg/dataset/Masks_for_Animal_data_Zhao'
    imgs_data  = {}
    files    = []
    Max_Shift = 50
    Is_Flip = True
    iCnt = 0
    
    imgs_folder = os.listdir(osp.join(data_dir, name))
    for file_folder_name in imgs_folder:
        imgs = os.listdir(osp.join(osp.join(data_dir, name), file_folder_name))
        for filename in imgs:
            imgs_data[filename] = sio.loadmat(osp.join(osp.join(osp.join(data_dir, name), file_folder_name),filename))['bMode']
            files.append({
                "img": None,
                "label": iCnt,
                "name": (osp.join(osp.join(osp.join(data_dir, name), file_folder_name),filename))
                })
            iCnt = iCnt + 1
    
    index = 5
    datafiles = files[index]  
    image_dir = datafiles["name"]
    image = sio.loadmat(image_dir)['bMode']
    
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
    '''
    
    #---CardiacLabelDataSet
    vis = Visualizer('GAN_Segmentation_DataLoader')
    
    #---CardiacDataSet
    files = []
    image_size = (128, 128, 128)
    curFolderNum = 0 
    folders = ['DSEC9BrfDOBSTR', 'DSEC9 baseline']
    data_dir = '/home/dragon/Downloads/AdvSemiSeg/dataset/Masks_for_Animal_data_Zhao'
    gt_data = {}
    for name in folders:
        gt_dir = osp.join(data_dir, name + '.mat')
        gt_data[name] = sio.loadmat(gt_dir)['bMode']
        imgs = os.listdir(osp.join(data_dir, name))
        for filename in imgs:
            img_file = osp.join(osp.join(data_dir, name), filename)
            label_file = int(re.findall(r"\d+d*",filename)[0])-1
            files.append({
                "img": img_file,
                "label": label_file,
                "name": name
                })
    index = 1
    datafiles = files[index]
    label = sio.loadmat(datafiles["img"])['myomask_revserse']
    image = gt_data[datafiles["name"]][:,:,:,datafiles["label"]]
    Is_Flip = True
    Max_Shift = 50
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
    image = t.nn.functional.grid_sample(t.from_numpy(image).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear', align_corners=False)
    label = t.nn.functional.grid_sample(t.from_numpy(label).float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),mode = 'bilinear', align_corners=False)
    vis.heatmap((image[0,0,32,:,:]), win='Input')
    vis.heatmap((label[0,0,32,:,:]), win='Label')
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    