'''
Main script for evaluating a DL model for mmWave beam prediction
'''

import os
import datetime
import shutil

import torch as t
import torch.cuda as cuda
import torch.optim as optimizer
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transf
from torchsummary import summary

import numpy as np
import pandas as pd

from data_feed import DataFeed
from build_net import resnet50

val_batch_size = 1
train_size = [1]
lr = 1e-4
decay = 1e-4


model = resnet50(pretrained=True, progress=True, num_classes=64)
model = model.cuda()


checkpoint_path = 'saved_folder/01-25-2023_17_39/checkpoint/resnet50_32_beam'
model.load_state_dict(t.load(checkpoint_path))
model.eval()



########################################################################
########################### Data pre-processing ########################
########################################################################
img_resize = transf.Resize((224, 224))

img_norm = transf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
proc_pipe = transf.Compose(
    [transf.ToPILImage(),
     img_resize,
     transf.ToTensor(),
     img_norm]
    )
val_dir = './scenario23_img_beam_test.csv'
val_loader = DataLoader(DataFeed(val_dir, transform=proc_pipe),
                        batch_size=val_batch_size,
                        #num_workers=8,
                        shuffle=False)
criterion = nn.CrossEntropyLoss()
opt = optimizer.Adam(model.parameters(), lr=lr, weight_decay=decay)  
val_acc = []   
feature_vec = []   
                
#with cuda.device(0):

top_1 = np.zeros( (1,len(train_size)) )
top_2 = np.zeros( (1,len(train_size)) )
top_3 = np.zeros( (1,len(train_size)) )
top_5 = np.zeros( (1,len(train_size)) )

running_top1_acc = []
running_top2_acc = []
running_top3_acc = []
running_top5_acc = []

print('Start validation')
ave_top1_acc = 0
ave_top2_acc = 0
ave_top3_acc = 0
ave_top5_acc = 0
ind_ten = t.as_tensor([0, 1, 2, 3, 4], device='cuda:0')
top1_pred_out = []
top2_pred_out = []
top3_pred_out = []
top5_pred_out = []
gt_beam = []
total_count = 0
for val_count, (imgs, labels) in enumerate(val_loader):
    model.eval()
    x = imgs.cuda()
    opt.zero_grad()
    labels = labels.cuda()
    total_count += labels.size(0)
    _, out = model.forward(x)
    _, top_1_pred = t.max(out, dim=1)
    
    gt_beam.append(labels.detach().cpu().numpy()[0])
    
    top1_pred_out.append(top_1_pred.detach().cpu().numpy()[0])
    sorted_out = t.argsort(out, dim=1, descending=True)
    
    top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
    top2_pred_out.append(top_2_pred.detach().cpu().numpy()[0])
    
    top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:3])
    top3_pred_out.append(top_3_pred.detach().cpu().numpy()[0])
    
    top_5_pred = t.index_select(sorted_out, dim=1, index=ind_ten)
    top5_pred_out.append(top_5_pred.detach().cpu().numpy()[0])                      
    
    reshaped_labels = labels.reshape((labels.shape[0], 1))
    tiled_2_labels = reshaped_labels.repeat(1, 2)
    tiled_3_labels = reshaped_labels.repeat(1, 3)
    tiled_5_labels = reshaped_labels.repeat(1, 5) 
    
    batch_top1_acc = t.sum(top_1_pred == labels, dtype=t.float32)
    batch_top2_acc = t.sum(top_2_pred == tiled_2_labels, dtype=t.float32)
    batch_top3_acc = t.sum(top_3_pred == tiled_3_labels, dtype=t.float32)
    batch_top5_acc = t.sum(top_5_pred == tiled_5_labels, dtype=t.float32)                    

    ave_top1_acc += batch_top1_acc.item()
    ave_top2_acc += batch_top2_acc.item()
    ave_top3_acc += batch_top3_acc.item()
    ave_top5_acc += batch_top5_acc.item()                    
print("total test examples are", total_count)
running_top1_acc.append(ave_top1_acc / total_count)  # (batch_size * (count_2 + 1)) )
running_top2_acc.append(ave_top2_acc / total_count)
running_top3_acc.append(ave_top3_acc / total_count)  # (batch_size * (count_2 + 1)))
running_top5_acc.append(ave_top5_acc / total_count)  # (batch_size * (count_2 + 1)))                

print('Average Top-1 accuracy {}'.format( running_top1_acc[-1]))
print('Average Top-2 accuracy {}'.format( running_top2_acc[-1]))
print('Average Top-3 accuracy {}'.format( running_top3_acc[-1]))
print('Average Top-5 accuracy {}'.format( running_top5_acc[-1])) 



 
 
 
 
 


