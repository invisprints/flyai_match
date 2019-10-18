# -*- coding: utf-8 -*
from flyai.processor.base import Base
from flyai.processor.download import check_download
from path import DATA_PATH
from PIL import Image
from flyai.processor.base import Base
import cv2
from path import DATA_PATH
import os
import numpy as np
from albumentations import (
    OneOf, Compose, Resize, RandomCrop, Flip, RandomRotate90, HueSaturationValue, GaussNoise,
    Rotate, Blur, Normalize, CenterCrop, RandomBrightnessContrast, Rotate, RGBShift, HorizontalFlip,
    ShiftScaleRotate, CoarseDropout
    )

img_size = (256, 256)
crop_size = (224, 224)
means = np.array([0.485, 0.456, 0.406])
stds = np.array([0.229, 0.224, 0.225])
train_aug =  Compose([
    Resize(img_size[0], img_size[1]), RandomCrop(crop_size[0], crop_size[1]),
    HorizontalFlip(p=0.5),  RandomBrightnessContrast(0.3, 0.3), ShiftScaleRotate(0.1), CoarseDropout(8, 32, 32),
    Normalize(means, stds)
])
valid_aug = Compose([
    Resize(img_size[0], img_size[1]), CenterCrop(crop_size[0], crop_size[1]),
    Normalize(means, stds)
])
'''
把样例项目中的processor.py件复制过来替换即可
'''

class Processor(Base):

    def __init__(self):
        self.img_shape = [crop_size, crop_size, 3]

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()和dataset.next_validation_batch()多次调用。
    '''
    def input_x(self, image_path):
        img = cv2.imread(os.path.join(DATA_PATH, image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = train_aug(image=img)['image']
        img = img.transpose((2, 0, 1))  # channel first
        return img

    '''
    参数为csv中作为输入x的一条数据，该方法会在评估时多次调用。
    '''

    def output_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = valid_aug(image=img)['image']
        img = img.transpose((2, 0, 1))  # channel first
        return img

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_y(self, labels):
        # one_hot_label = numpy.zeros([10])  ##生成全0矩阵
        # one_hot_label[labels] = 1  ##相应标签位置置
        # return one_hot_label
        return labels

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        # return numpy.argmax(data)
        return data