# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset
from model import Model
import numpy as np

data = Dataset()
model = Model(data)


dataset = Dataset()
x_test, y_test = dataset.evaluate_data_no_processor('dev.csv')
preds = model.predict_all(x_test)
print(preds)
# print(y_test['labels'])
labels = [x['labels'] for x in y_test]
print(labels)
print(np.sum(np.array(preds)==np.array(labels))/len(labels))
###########################################

# data = x_test[5]
# preds = model.predict(**data)
# print(preds)
# label = y_test[5]['labels']
# target = model.data.to_categorys(label)
# print(target)
