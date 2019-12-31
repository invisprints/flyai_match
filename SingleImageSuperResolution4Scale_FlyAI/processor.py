# -*- coding: utf-8 -*

import numpy
from flyai.processor.base import Base
import os
import cv2
from path import DATA_PATH
import numpy as np

'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''
    def input_x(self, lr_image_path):
        image_path = os.path.join(DATA_PATH, lr_image_path)
        return image_path

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''
    def input_y(self, hr_image_path):
        label_path = os.path.join(DATA_PATH, hr_image_path)
        return label_path

    '''
    参数为csv中作为输入x的一条数据，该方c法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''
    def output_y(self, pred_label):
        # pred_label = np.squeeze(pred_label)
        return pred_label
