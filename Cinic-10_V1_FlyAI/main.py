# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import torch
from flyai.dataset import Dataset
from model import Model
from path import MODEL_PATH
import torchvision
from torch import nn
import torch.optim as optim
from flyai.utils.log_helper import train_log
from torch.optim.lr_scheduler import *
import numpy as np

from torch.utils.data import DataLoader
from Cinicdataset import CinicData

from albumentations import (
    OneOf, Compose, Resize, RandomCrop, Normalize, CenterCrop, RandomBrightnessContrast,
    HorizontalFlip, ShiftScaleRotate, CoarseDropout
)
'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

x_train, y_train, x_val, y_val = dataset.get_all_data()
def create_dataset(train):
    img_size = (256, 256)
    crop_size = (224, 224)
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    if train:
        train_aug =  Compose([
            Resize(img_size[0], img_size[1]), RandomCrop(crop_size[0], crop_size[1]),
            HorizontalFlip(p=0.5),  RandomBrightnessContrast(0.3, 0.3), ShiftScaleRotate(0.1), CoarseDropout(8, 32, 32),
            Normalize(means, stds)
        ])
        dataset = CinicData(x_train, y_train, transformation=train_aug)
    else:
        valid_aug = Compose([
            Resize(img_size[0], img_size[1]), CenterCrop(crop_size[0], crop_size[1]),
            Normalize(means, stds)
        ])
        dataset = CinicData(x_val, y_val, transformation=valid_aug)
    return dataset

train_dataloader =  DataLoader(create_dataset(True), shuffle=True, batch_size=args.BATCH)

valid_dataloader =  DataLoader(create_dataset(False), shuffle=False, batch_size=args.BATCH)

'''
实现自己的网络机构
'''
# 判断gpu是否可用
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_class = 10

def get_resnet101(n_class):
    net = torchvision.models.resnet101(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, n_class)
    return net

def get_densenet201(n_class):
    net = torchvision.models.densenet201(pretrained=True)
    num_ftrs = net.classifier.in_features
    net.classifier = nn.Linear(num_ftrs, n_class)
    return net

net = get_resnet101(n_class)
# net = get_densenet201(n_class)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min')

for epo in range(args.EPOCHS):
    print('-----{}/{}-----'.format(epo, args.EPOCHS))
    net.train()
    for x_train, y_train in train_dataloader:
        x_train, y_train = x_train.float().to(device), y_train.to(device)
        optimizer.zero_grad()
        outputs = net(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        outputs = outputs.argmax(1).cpu().numpy()
        y_train = y_train.cpu().numpy()
        acc = np.sum(y_train == outputs)/len(y_train)
        train_log(train_loss=loss.item(),train_acc=acc)

    net.eval()
    avg_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for x_val, y_val in valid_dataloader:
            x_val, y_val = x_val.float().to(device), y_val.to(device)
            outputs = net(x_val)
            loss = criterion(outputs, y_val)

            avg_loss += loss.item()
            outputs = outputs.argmax(1).cpu().numpy()
            y_val = y_val.cpu().numpy()
            acc = np.sum(y_val == outputs)/len(y_val)
            val_acc += np.sum(y_val == outputs)/len(y_val)
            train_log(val_loss=loss.item(),val_acc=acc)
    avg_loss /= len(valid_dataloader)
    val_acc /= len(valid_dataloader)

    scheduler.step(avg_loss)
    print('lr=%f'%(optimizer.param_groups[0]['lr']))
    print('avg_loss=%f,val_acc=%f'%(avg_loss,val_acc))
model.save_model(net, MODEL_PATH, overwrite=True)
print("Train Finished")

