# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017
@author: user
"""
import argparse
import torch
import torch.nn as nn
from flyai.dataset import Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import *
import numpy as np
from model import Model
from path import MODEL_PATH
from flyai.utils.log_helper import train_log
from torch.utils.data import DataLoader
from torchvision.models.segmentation import *
from segdataset import SkyData
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    Compose,
    RandomRotate90,
    RandomResizedCrop,
    RandomBrightnessContrast,
    Normalize
)

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=3, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
data = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(data)
mean, std = model.get_mean_std()
resize = 600
crop_size = 520
train_aug = Compose([
    RandomResizedCrop(height=crop_size, width=crop_size, p=1),
    VerticalFlip(p=0.5),
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.5),
    RandomBrightnessContrast(p=0.3),
    Normalize(mean, std)
    ])
valid_aug = Compose([
    Resize(resize, resize),
    CenterCrop(crop_size,crop_size),
    VerticalFlip(p=0.3),
    HorizontalFlip(p=0.3),
    RandomRotate90(p=0.3),
    RandomBrightnessContrast(p=0.3),
    Normalize(mean, std)
])


x_train, y_train, x_val, y_val = data.get_all_data()
print(mean, std)
train_dataset = SkyData(x_train, y_train, transformation=train_aug)
valid_dataset = SkyData(x_val, y_val, transformation=valid_aug)
train_dataloader = DataLoader(train_dataset,batch_size=args.BATCH,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=args.BATCH)

# 判断gpu是否可用
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''
实现自己的网络结构
'''
# cnn =  deeplabv3_resnet101(pretrained=False, num_classes=2)
cnn =  fcn_resnet101(pretrained=False, num_classes=2)
cnn = cnn.to(device)
optimizer = SGD(cnn.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
# scheduler = StepLR(optimizer,step_size=7,gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min')

# criterion = nn.BCELoss()  # 定义损失函数
def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        # x.transpose_(1,2)
        # x.transpose_(2,3)
        loss_fn = nn.CrossEntropyLoss()
        losses[name] = loss_fn(x, target)

    return losses['out']
'''
dataset.get_step() 获取数据的总迭代次数
'''

for epo in range(args.EPOCHS):
    cnn.train()
    for x_train,y_train in train_dataloader:
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        outputs = cnn(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_log(train_loss=loss.item())
    # val
    cnn.eval()
    loss_m = 0.0
    for x_val, y_val in valid_dataloader:
        with torch.no_grad():
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = cnn(x_val)
            val_loss = criterion(outputs, y_val)
            train_log(val_loss=val_loss.item())
            loss_m += val_loss.item()
    loss_m /= len(valid_dataloader)
    print("val_loss=%f"%loss_m)

    print("lr=%f"%optimizer.param_groups[0]["lr"])
    scheduler.step(loss_m)

model.save_model(cnn, MODEL_PATH, overwrite=True)
print("saved model!!!")
