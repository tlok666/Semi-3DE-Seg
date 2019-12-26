#--Author: Long Teng
#--Data 2019-12
import os
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import os.path as osp
import scipy.io as sio
from torch.utils import data
from model.unet import UNet
from dataset.cardiac_dataset import CardiacDataSet3D, CardiacDataSet3D_Gt, CardiacNoLabelDataSet_3D
from dataset.Synthetic_dataset import SyntheticDataSet3D, SyntheticDataSet3D_Gt, SyntheticNoLabelDataSet3D
from utils.visualize import Visualizer
from torchnet.meter import AverageValueMeter


parser = argparse.ArgumentParser()
parser.add_argument("--vis_name", type=str, default='GAN_Segmentation',help="Where to save snapshots of the model.")
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--num_classes", type=int, default=1, help="Number of classes to predict (including background).")
parser.add_argument("--lambda_adv", type=float, default=0.001, help="lambda_adv for semi-supervised training.")
parser.add_argument("--lambda_self", type=float, default=0.001, help="lambda_self for self-supervised training.")
parser.add_argument("--snapshot_dir", type=str, default='./snapshots/',help="Where to save snapshots of the model.")
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
    train_dataset = CardiacDataSet3D(crop_size=input_size)
    train_remain_dataset = CardiacNoLabelDataSet_3D(crop_size=input_size)
    train_gt_dataset = CardiacDataSet3D_Gt(crop_size=input_size)
else:
    train_dataset = SyntheticDataSet3D(crop_size=input_size)
    train_remain_dataset = SyntheticNoLabelDataSet3D(crop_size=input_size)
    train_gt_dataset = SyntheticDataSet3D_Gt(crop_size=input_size)

trainloader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
trainloader_remain = data.DataLoader(train_remain_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
trainloader_gt = data.DataLoader(train_gt_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        
trainloader_remain_iter = enumerate(trainloader_remain)
trainloader_gt_iter = enumerate(trainloader_gt)


# ----------
#  Training
# ----------
DType = opt.data_type  #-- 0, 1
LType = opt.loss_type  #-- 0, 1, 2
data_type = ['Synthetic', 'Clinical']
loss_type = ['baseline', 'adv', 'adv_semi']
if opt.resume_epoch != 0:
    load_path = osp.join(osp.join(opt.snapshot_dir, data_type[DType]), loss_type[LType])
    if os.path.exists(osp.join(load_path, 'Cardiac_'+ str(opt.resume_epoch) +'.pth')):
        generator.load_state_dict(torch.load(osp.join(load_path, 'Cardiac_'+ str(opt.resume_epoch) +'.pth')))
        print('Successed in loading the model G...')
    if os.path.exists(osp.join(load_path, 'Cardiac_'+ str(opt.resume_epoch) +'_D.pth')):
        discriminator.load_state_dict(torch.load(osp.join(load_path, 'Cardiac_'+ str(opt.resume_epoch) +'_D.pth')))
        print('Successed in loading the model D...')
# ----------
#  Training
# ----------
vis = Visualizer(opt.vis_name)
error_loss_adv_G  = AverageValueMeter()
error_loss_adv_D  = AverageValueMeter()
error_loss_self = AverageValueMeter()
error_loss_seg  = AverageValueMeter()

loss_curve_pre_seg = []
loss_curve_mask_T = []

for epoch in range(opt.resume_epoch, opt.n_epochs):
    for i, (imgs, labels, _, cc) in enumerate(trainloader):
        #----DataLoader
        try:
            _, batch_unlabled = next(trainloader_remain_iter)
        except:
            trainloader_remain_iter = enumerate(trainloader_remain)
            _, batch_gt = trainloader_remain_iter.__next__()
        
        try:
            _, batch_gt       = next(trainloader_gt_iter)
        except:
            trainloader_iter = enumerate(trainloader_gt)
            _, batch_gt = trainloader_iter.__next__()
            
        images_unlabeled, _, _ = batch_unlabled
        _, labels_gt, _, _ = batch_gt
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(imgs.shape).fill_(0.0), requires_grad=False).cuda()
        # Configure input
        real_imgs = Variable(labels.float()).cuda()
        labels_gt = Variable(labels_gt.float()).cuda()
        #print(cc)
        
        
        
        
        # -----------------
        #  Semi-supervised Adv Loss
        # -----------------
        # -----------------
        #  Train Generator
        # -----------------
        if LType != 0:
            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = Variable(imgs).cuda()
            z_extra = Variable(images_unlabeled).cuda()
            # Generate a batch of images
            gen_imgs = generator(z)
            gen_imgs_extra = generator(z_extra)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss = opt.lambda_adv * g_loss
            g_loss.backward()
            g_loss = adversarial_loss(discriminator(gen_imgs_extra), valid)
            g_loss = opt.lambda_adv * g_loss
            g_loss.backward()
            optimizer_G.step()
            error_loss_adv_G.add(g_loss.detach().cpu().numpy())
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid) + adversarial_loss(discriminator(labels_gt), valid)
            real_loss = opt.lambda_adv *real_loss
            real_loss.backward()
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake) + adversarial_loss(discriminator(gen_imgs_extra.detach()), fake)
            fake_loss = opt.lambda_adv * fake_loss
            fake_loss.backward()
            optimizer_D.step()
            d_loss = real_loss + fake_loss
            error_loss_adv_D.add(d_loss.detach().cpu().numpy())
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(trainloader), d_loss.item(), g_loss.item())
            )
            batches_done = epoch * len(trainloader) + i
        
        
        
        
        # -----------------
        #  Self-supervised Seg Loss
        # -----------------
        if LType == 2:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            gen_imgs_extra = generator(z_extra)
            D_out_sigmoid = discriminator(gen_imgs_extra).detach()
            semi_gt = (D_out_sigmoid > discriminator.mask_T).float()
                    
            loss_self = loss_calc(gen_imgs_extra, semi_gt)
            loss_self = opt.lambda_self * loss_self
            loss_self.backward()
            error_loss_self.add(loss_self.detach().cpu().numpy())
            optimizer_G.step()
            optimizer_D.step()
        
        
        # -----------------
        #  Supervised Seg Loss
        # -----------------
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        z = Variable(imgs).cuda()
        gen_imgs = generator(z)
        loss_seg = loss_calc(gen_imgs, real_imgs)
        loss_seg = 1 * loss_seg
        loss_seg.backward()
        error_loss_seg.add(loss_seg.detach().cpu().numpy())
        optimizer_G.step()
        optimizer_D.step()
        # -----------------
        #  Visualize
        # -----------------
        loss_curve_pre_seg.append(loss_seg.detach().cpu().numpy())
        loss_curve_mask_T.append(discriminator.mask_T.detach().cpu().numpy())
        
        vis.plot('Loss_Seg', error_loss_seg.value()[0])
        vis.plot('Loss_Self', error_loss_self.value()[0])
        vis.plot('Loss_Adv_D', error_loss_adv_D.value()[0])
        vis.plot('Loss_Adv_G', error_loss_adv_G.value()[0])
        vis.plot('loss_curve_mask_T', discriminator.mask_T.detach().cpu().numpy())
        
        vis.heatmap((gen_imgs.data.cpu().numpy()[0,0,32,:,:]), win='Pred')
        vis.heatmap((labels.data.cpu().numpy()[0,0,32,:,:]), win='Gt')
        vis.heatmap((imgs.data.cpu().numpy()[0,0,32,:,:]*128.0+127.0), win='Input')
        
        
        
        
    save_path = osp.join(osp.join(opt.snapshot_dir, data_type[DType]), loss_type[LType])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if epoch % opt.sample_interval == 0:
        print('taking snapshot ...')
        torch.save(generator.state_dict(),osp.join(save_path, 'Cardiac_'+str(epoch)+'.pth'))
        torch.save(discriminator.state_dict(),osp.join(save_path, 'Cardiac_'+str(epoch)+'_D.pth'))
        
        torch.save(generator.state_dict(),osp.join(save_path, 'Cardiac_'+'latest'+'.pth'))
        torch.save(discriminator.state_dict(),osp.join(save_path, 'Cardiac_'+'latest'+'_D.pth'))

save_path = osp.join(osp.join(opt.snapshot_dir, data_type[DType]), loss_type[LType])
if not os.path.exists(save_path):
    os.makedirs(save_path)
sio.savemat(osp.join(save_path, 'loss_curve_seg.mat'), {'loss_curve':loss_curve_pre_seg})
sio.savemat(osp.join(save_path, 'loss_curve_mask_T.mat'), {'loss_curve':loss_curve_mask_T})
        
        
        
        
        
        
        
        
        
        
