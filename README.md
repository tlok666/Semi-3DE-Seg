# Semi-supervised-Echocardiography-Segmentation
基于对抗网络的三维超声心动图左心室心肌半监督分割方法


Envirment require:
1. Pytorch=1.4.1
2. visdom
3. Python3.6

Training:
1. python -m visdom.server
2. python tran.py --data_type==0 --loss_type=0 --resume_epoches=0 --vis_name='GAN_Seg'
   --data_type: Synthetic data(0),  Animal data(1)
   --loss_type: Supervised(0), Semi-supervised(1), Semi-supervised+Self-supervised(2)
   --resume_epoches: Previous training *pth
   --vis_name: visdom view name
   
  
Testing:
1. python -m visdom.server
2. python test.py --data_type==0  --loss_type=0  --resume_epoches=0 --vis_name='GAN_Test_Seg'
   --data_type: Synthetic data(0),  Animal data(1)
   --loss_type: Supervised(0), Semi-supervised(1), Semi-supervised+Self-supervised(2)
   --resume_epoches: Previous training *pth
   --vis_name: visdom view name
   
During training or testing, open localhost:8097 to see the visulized variable.
