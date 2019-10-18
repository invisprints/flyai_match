# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from PIL import Image
from flyai.dataset import Dataset
from model import Model
import matplotlib.pyplot as plt
from path import DATA_PATH
from os.path import join
import numpy as np

data = Dataset()
model = Model(data)

dataset = Dataset()
x_test, y_test = dataset.evaluate_data_no_processor('dev.csv')
# preds = model.predict_all(x_test)
data = x_test[9]
preds = model.predict(**data)
print(preds.shape)
plt.imshow(preds)
plt.show()

path = join(DATA_PATH, y_test[9]['label_path'])
target = Image.open(path)
target = np.array(target)
target = target[...,0]
target = model.data.to_categorys(target)
print(target.shape)
plt.imshow(target)
plt.show()