# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
import cv2
from path import MODEL_PATH
import net
import utility
import numpy as np
from option import Config

TORCH_MODEL_NAME = "model.pkl"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        # self.config = Config()
        self.config = Config(resume='best')
        # if os.path.exists(self.net_path):
        #     self.net = torch.load(self.net_path)

    def predict(self, **data):
        checkpoint = utility.checkpoint(self.config)
        model = net.Model(self.config, checkpoint)
        # model.load(MODEL_PATH, resume='best')
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            x_data = self.data.predict_data(**data)
            x_data = cv2.imread(x_data[0])
            x_data = cv2.cvtColor(x_data, cv2.COLOR_BGR2RGB).transpose(2,0,1)
            x_data = torch.from_numpy(x_data).float()
            x_data = x_data.to(device)[None]
            sr = model(x_data, 0)
            sr = utility.quantize(sr, 255)
            sr = torch.squeeze(sr.cpu())
            prediction = sr.numpy().transpose(1,2,0)
            prediction = np.array(prediction, dtype=np.uint8)
        return prediction

    def predict_all(self, datas):
        checkpoint = utility.checkpoint(self.config)
        model = net.Model(self.config, checkpoint)
        model.load(MODEL_PATH)
        model = model.to(device)
        model.eval()
        labels = []
        with torch.no_grad():
            for data in datas:
                x_data = self.data.predict_data(**data)
                x_data = cv2.imread(x_data[0])
                x_data = cv2.cvtColor(x_data, cv2.COLOR_BGR2RGB).transpose(2,0,1)
                x_data = torch.from_numpy(x_data).float()
                x_data = x_data.to(device)[None]
                sr = model(x_data, 0)
                sr = utility.quantize(sr, 255)
                sr = torch.squeeze(sr.cpu())
                prediction = sr.numpy().transpose(1,2,0)
                prediction = np.array(prediction, dtype=np.uint8)
                labels.append(prediction)

        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))
