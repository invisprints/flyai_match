# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import argparse
import torch
from flyai.dataset import Dataset
from torch.utils.data import DataLoader
from model import Model
from net import get_model
from path import MODEL_PATH
import numpy as np
from hatdataset import HatData
import utils
from flyai.utils.log_helper import train_log
import time
from albumentations import (
    BboxParams,
    HorizontalFlip,
    RandomSizedBBoxSafeCrop,
    Crop,
    Compose
)

from lib.Evaluator import *
from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.utils import *

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area,
                                               min_visibility=min_visibility, label_fields=['labels']))
def get_transform(train):
    if train:
        return get_aug([HorizontalFlip(p=0.5),RandomSizedBBoxSafeCrop(height=256,width=256)])
    else:
        return get_aug([HorizontalFlip(p=0.5)])

train_aug = get_transform(True)
valid_aug = get_transform(False)
'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=2, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

x_train, y_train, x_val, y_val = dataset.get_all_data()
# x_train, y_train, x_val, y_val = dataset.get_all_processor_data()
train_dataset = HatData(x_train, y_train, transformation=train_aug)
valid_dataset = HatData(x_val, y_val, transformation=valid_aug)
train_dataloader = DataLoader(train_dataset,batch_size=args.BATCH,shuffle=True,collate_fn=utils.collate_fn)
valid_dataloader = DataLoader(valid_dataset,batch_size=args.BATCH,collate_fn=utils.collate_fn)

'''
实现自己的网络机构
'''
# 判断gpu是否可用
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net = get_model(3, True)
net.to(device)

# construct an optimizer
params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)

'''
dataset.get_step() 获取数据的总迭代次数

'''
datalen = len(train_dataloader)
maxap = 0.
for epo in range(args.EPOCHS):
    net.train()
    for batch, (imgs, targets) in enumerate(train_dataloader):
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = net(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step(epo + batch/datalen)
        train_log(train_loss=losses.item(), val_acc=lr_scheduler.get_lr()[0])

    # start_time = time.time()

    if epo < 5:
        continue
    net.eval()
    myBoundingBoxes = BoundingBoxes()
    with torch.no_grad():
        for imgs, targets in valid_dataloader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = net(imgs)
            outputs = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in outputs]
            targets = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in targets]
            for idx in range(len(targets)):
                target = targets[idx]
                for i in range(len(target['labels'])):
                    box = target['boxes'][i]
                    myBoundingBoxes.addBoundingBox(BoundingBox(
                        imageName=target['image_id'],
                        classId=target['labels'][i],
                        x=box[0],y=box[1],
                        w=box[2],h=box[3],
                        format=BBFormat.XYX2Y2,
                        bbType=BBType.GroundTruth
                    ))
                output = outputs[idx]
                for i in range(len(output['labels'])):
                    myBoundingBoxes.addBoundingBox(BoundingBox(
                        imageName=target['image_id'],
                        classId=output['labels'][i],
                        classConfidence=output['scores'][i],
                        x = output['boxes'][i][0],
                        y = output['boxes'][i][1],
                        w = output['boxes'][i][2],
                        h = output['boxes'][i][3],
                        bbType=BBType.Detected,
                        format=BBFormat.XYX2Y2
                    ))

    # print('compose box time:', time.time()-start_time)
    avg_map = 0.0
    evaluator = Evaluator()
    metricsPerClass = evaluator.GetPascalVOCMetrics(
        myBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=0.5,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code

    print("Average precision values per class:\n")
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        avg_map += average_precision
        # Print AP per class
        print('%s: %f' % (c, average_precision))

    # print('eval time:', time.time()-start_time)

    avg_map /= len(metricsPerClass)
    if maxap < avg_map:
        maxap = avg_map
        model.save_model(net, MODEL_PATH, overwrite=True)
        print(avg_map)


