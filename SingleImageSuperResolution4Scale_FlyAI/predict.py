# -*- coding: utf-8 -*
'''
实现模型的调用
'''

from flyai.dataset import Dataset
from model import Model
import matplotlib.pyplot as plt
from os.path import join
from path import DATA_PATH
import numpy as np
import cv2

data = Dataset()
model = Model(data)

dataset = Dataset()
x_test, y_test = dataset.evaluate_data_no_processor('dev.csv')
# preds = model.predict_all(x_test)
path = join(DATA_PATH, x_test[9]['lr_image_path'])
x_data = cv2.imread(path)
target = cv2.cvtColor(x_data, cv2.COLOR_BGR2RGB)
target = model.data.to_categorys(target)
plt.imshow(target)
plt.show()

data = x_test[9]
preds = model.predict(**data)
plt.imshow(preds)
plt.show()
# cv2.imshow('pred', preds)
# cv2.waitKey(0)

path = join(DATA_PATH, y_test[9]['hr_image_path'])
x_data = cv2.imread(path)
target = cv2.cvtColor(x_data, cv2.COLOR_BGR2RGB)
target = model.data.to_categorys(target)
plt.imshow(target)
plt.show()