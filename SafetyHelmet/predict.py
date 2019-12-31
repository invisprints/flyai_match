# -*- coding: utf-8 -*
'''
实现模型的调用
'''

from flyai.dataset import Dataset
from PIL import Image
from model import Model
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from path import DATA_PATH
from hatdataset import visualize_bbox
from lib.Evaluator import *

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.utils import *

data = Dataset()
model = Model(data)

dataset = Dataset()
x_test, y_test = dataset.evaluate_data_no_processor('dev.csv')
# 用于测试 predict_all 函数
myBoundingBoxes = BoundingBoxes()
preds = model.predict_all(x_test)
idx = 5
img = cv2.imread(join(DATA_PATH, x_test[idx]['img_path']))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
data = x_test[idx]
preds = model.predict(**data)
for i in range(len(preds['labels'])):
   myBoundingBoxes.addBoundingBox(BoundingBox(
       imageName=idx,
       classId=preds['labels'][i],
       classConfidence=preds['scores'][i],
       x = preds['boxes'][i][0],
       y = preds['boxes'][i][1],
       w = preds['boxes'][i][2],
       h = preds['boxes'][i][3],
       bbType=BBType.Detected,
       format=BBFormat.XYX2Y2
   ))

#show label
box_list = y_test[idx]['box']
box_list = box_list.split(' ')
for i in range(len(box_list)):
    box = box_list[i]
    box = box.split(',')
    box[0], box[1], box[2], box[3], box[4] = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4])#pascal_voc format
    myBoundingBoxes.addBoundingBox(BoundingBox(
        imageName=idx,
        classId=box[4],
        x=box[0],y=box[1],
        w=box[2],h=box[3],
        format=BBFormat.XYX2Y2,
        bbType=BBType.GroundTruth
    ))

img = myBoundingBoxes.drawAllBoundingBoxes(img,idx)
plt.imshow(img)
plt.show()
evaluator = Evaluator()
# Get metrics with PASCAL VOC metrics
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
    # Print AP per class
    print('%s: %f' % (c, average_precision))