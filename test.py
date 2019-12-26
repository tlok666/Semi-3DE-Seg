#--Author: Long Teng
#--Data 2019-12
import re
import os
import argparse
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import os.path as osp
import scipy.io as sio
from torch.utils import data
from model.unet import UNet
from dataset.cardiac_dataset import CardiacTestDataSet
from dataset.Synthetic_dataset import SyntheticTestSet3D
from utils.visualize import Visualizer
from torchnet.meter import AverageValueMeter
from utils.loss import DICE, IOU


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--num_classes", type=int, default=1, help="Number of classes to predict (including background).")
parser.add_argument("--lambda_adv", type=float, default=0.001, help="lambda_adv for semi-supervised training.")
parser.add_argument("--lambda_self", type=float, default=0.1, help="lambda_self for self-supervised training.")
parser.add_argument("--snapshot_dir", type=str, default='./snapshots/',help="Where to save snapshots of the model.")
parser.add_argument("--results_dir", type=str, default='./results/',help="Where to save test result data.")
parser.add_argument("--data_type", type=int, default=1, help="[0:'Synthetic', 1:'Clinical'].")
parser.add_argument("--loss_type", type=int, default=0, help="[0:'baseline', 1:'adv', 2:'adv_semi'].")
parser.add_argument("--resume_epoch", type=int, default=0, help="[0:'baseline', 1:'adv', 2:'adv_semi'].")
opt = parser.parse_args()
print(opt)


# Data Size
input_size = ( opt.img_size,  opt.img_size,  opt.img_size)
cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = torch.nn.BCELoss()
def loss_calc(pred, label, gpu=0):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label).cuda(gpu)
    if opt.num_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    return criterion(pred, label)

# Initialize generator and discriminator
generator = UNet(n_channels=opt.channels, n_classes=opt.num_classes)
discriminator = UNet(n_channels=opt.channels, n_classes=opt.num_classes, IsDiscriminator=True)
    
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# DataLoader
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if opt.data_type == 1:
    test_dataset = CardiacTestDataSet(crop_size=input_size)
else:
    test_dataset = SyntheticTestSet3D(crop_size=input_size)
    
testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False) 
dice = DICE().cuda()
iou  = IOU().cuda()
# ----------
#  Training
# ----------
DType = opt.data_type  #-- 0, 1
LType = opt.loss_type  #-- 0, 1, 2
data_type = ['Synthetic', 'Clinical']
loss_type = ['baseline', 'adv', 'adv_semi']

load_path = osp.join(osp.join(opt.snapshot_dir, data_type[DType]), loss_type[LType])
if opt.resume_epoch != 0:
    load_path = osp.join(osp.join(opt.snapshot_dir, data_type[DType]), loss_type[LType])
    if os.path.exists(osp.join(load_path, 'Cardiac_'+ str(opt.resume_epoch) +'.pth')):
        generator.load_state_dict(torch.load(osp.join(load_path, 'Cardiac_'+ str(opt.resume_epoch) +'.pth')))
        print('Successed in loading the model G...')
    if os.path.exists(osp.join(load_path, 'Cardiac_'+ str(opt.resume_epoch) +'_D.pth')):
        discriminator.load_state_dict(torch.load(osp.join(load_path, 'Cardiac_'+ str(opt.resume_epoch) +'_D.pth')))
        print('Successed in loading the model D...')
else:
    if os.path.exists(osp.join(load_path, 'Cardiac_'+'latest'+'.pth')):
        generator.load_state_dict(torch.load(osp.join(load_path, 'Cardiac_'+'latest'+'.pth')))
        print('Successed in loading the model G...')
    if os.path.exists(osp.join(load_path, 'Cardiac_'+'latest'+'_D.pth')):
        discriminator.load_state_dict(torch.load(osp.join(load_path, 'Cardiac_'+'latest'+'_D.pth')))
        print('Successed in loading the model D...')
    
# ----------
#  Training
# ----------
loss_IOU_T  = []
loss_DICE_T = []
vis = Visualizer('GAN_Testing')

for i, (imgs, labels, _, name) in enumerate(testloader):
    # -----------------
    #  Segmentation
    # -----------------
    Threshold = 0.1
    
    optimizer_G.zero_grad()
    optimizer_D.zero_grad()
    z = Variable(imgs).cuda()
    gen_imgs = generator(z)
    gen_imgs = (gen_imgs > Threshold).float()
    
    vis.heatmap((gen_imgs.data.cpu().numpy()[0,0,32,:,:]), win='Pred')
    vis.heatmap((labels.data.cpu().numpy()[0,0,32,:,:]), win='Gt')
    vis.images((z.data.cpu().numpy()[0,0,32,:,:]), win='Image')
        
    dice_result = dice(gen_imgs, labels.cuda())
    iou_result = iou(gen_imgs, labels.cuda())  
    
    loss_DICE_T.append(dice_result.detach().cpu().numpy())
    loss_IOU_T.append(iou_result.detach().cpu().numpy())
    
    print(
            "[Epoch %d/%d] [Batch %d/%d] [IOU loss: %f] [DICE loss: %f]"
            % (1, 1, i, len(testloader), iou_result.detach().cpu().numpy(), dice_result.detach().cpu().numpy())
        )
    
    save_iCnt = (re.findall(r"\d+d*",name[0]))
    save_iCnt = int(save_iCnt[len(save_iCnt)-1])
    save_path = osp.join(osp.join(opt.results_dir, data_type[DType]), loss_type[LType])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sio.savemat(osp.join(save_path, 'mask'+ str(save_iCnt)+'.mat'), {'bMode':gen_imgs.data.detach().cpu().numpy()})
        
print("[IOU loss: %f]" %(np.sum(loss_IOU_T)/len(loss_IOU_T)))
print("[DICE loss: %f]" %(np.sum(loss_DICE_T)/len(loss_DICE_T)))
        
        
        
        
        
        
        
        
        
        
        
        
