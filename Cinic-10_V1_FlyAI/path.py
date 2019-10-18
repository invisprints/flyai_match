# -*- coding: utf-8 -*
import sys

import os

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')

data_dir = os.path.join(sys.path[0], 'data')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

output_dir = os.path.join(sys.path[0], 'data', 'output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
