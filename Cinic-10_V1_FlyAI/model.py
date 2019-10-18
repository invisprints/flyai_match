# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
from path import MODEL_PATH
from albumentations import (
    OneOf, Compose, Resize, RandomCrop, Normalize, CenterCrop
)

TORCH_MODEL_NAME = "model.pkl"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

img_size = (256, 256)
crop_size = (224, 224)
means = numpy.array([0.485, 0.456, 0.406])
stds = numpy.array([0.229, 0.224, 0.225])
valid_aug = Compose([
    Resize(img_size[0], img_size[1]), CenterCrop(crop_size[0], crop_size[1]),
    Normalize(means, stds)
])

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        if os.path.exists(self.net_path):
            self.net = torch.load(self.net_path)
            print("model load")

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.net_path)
        self.net.to(device)
        self.net.eval()
        with torch.no_grad():
            x_data = self.data.predict_data(**data)
            x_data = torch.from_numpy(x_data)
            x_data = x_data.float().to(device)
            outputs = self.net(x_data)
            _, prediction = torch.max(outputs.data, 1)
            prediction = prediction.cpu().item()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.net_path)
            print("model load")
        self.net.to(device)
        self.net.eval()
        labels = []
        with torch.no_grad():
            for data in datas:
                x_data = self.data.predict_data(**data)
                x_data = torch.from_numpy(x_data)
                x_data = x_data.float().to(device)
                outputs = self.net(x_data)
                _, prediction = torch.max(outputs.data, 1)
                prediction = prediction.cpu().item()
                prediction = self.data.to_categorys(prediction)
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
