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
from mydataset import MyData
from torch.utils.data import DataLoader
import net
import utility
from option import Config
import loss
from trainer import Trainer

'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

config = Config(epochs=args.EPOCHS, batch_size=args.BATCH, pre_train='download')
checkpoint = utility.checkpoint(config)
'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=config.epochs, batch=config.batch_size)
model = Model(dataset)
x_train, y_train, x_val, y_val = dataset.get_all_data()
# x_train, y_train, x_val, y_val = dataset.get_all_processor_data()
train_dataset = MyData(x_train, y_train, transformation='train')
valid_dataset = MyData(x_val, y_val, transformation='valid')
train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=config.batch_size)
'''
实现自己的网络机构
'''
# 判断gpu是否可用
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
_model = net.Model(config, checkpoint)
_loss = loss.Loss(config, checkpoint)
t = Trainer(config, train_dataloader, valid_dataloader, _model, _loss, checkpoint)

'''
dataset.get_step() 获取数据的总迭代次数
'''

while not t.terminate():
    t.train()
    t.test()

checkpoint.done()

# best_score = 0
# for step in range(dataset.get_step()):
#     x_train, y_train = dataset.next_train_batch()
#     x_val, y_val = dataset.next_validation_batch()
#     '''
#     实现自己的模型保存逻辑
#     '''
#     model.save_model(net, MODEL_PATH, overwrite=True)
#     print(str(step + 1) + "/" + str(dataset.get_step()))
