# -*- coding: utf-8 -*
import numpy
import os
import torch
from net import get_model
from flyai.model.base import Base
from PIL import Image
import torchvision.transforms.functional as F

from path import MODEL_PATH

TORCH_MODEL_NAME = "model.pkl"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        if os.path.exists(self.net_path):
            self.net = get_model(3, True)
            self.net.load_state_dict(torch.load(self.net_path))

    def predict(self, **data):
        self.net.to(device)
        self.net.eval()
        x_data = self.data.predict_data(**data)
        x_data = Image.open(x_data[0])
        with torch.no_grad():
            x_data = F.to_tensor(x_data)
            x_data = x_data.to(device)
            outputs = self.net([x_data])
            outputs = outputs[0]
            outputs['labels'] = outputs['labels'].cpu().numpy() - 1
            outputs['boxes'] = outputs['boxes'].cpu().numpy()
            outputs['scores'] = outputs['scores'].cpu().numpy()
            prediction = outputs
        return prediction

    def predict_all(self, datas):
        labels = []
        self.net.to(device)
        self.net.eval()
        for data in datas:
            x_data = self.data.predict_data(**data)
            x_data = Image.open(x_data[0])
            with torch.no_grad():
                x_data = F.to_tensor(x_data)
                x_data = x_data.to(device)
                outputs = self.net([x_data])
                outputs = outputs[0]
                outputs['labels'] = outputs['labels'].cpu().numpy() - 1
                outputs['boxes'] = outputs['boxes'].cpu().numpy()
            prediction = []
            for i in range(len(outputs['labels'])):
                prediction.append([outputs['labels'][i], outputs['boxes'][i][0],
                                  outputs['boxes'][i][1], outputs['boxes'][i][2],outputs['boxes'][i][3]])
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
        torch.save(network.state_dict(), os.path.join(path, name))
