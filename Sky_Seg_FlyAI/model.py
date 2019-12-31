# -*- coding: utf-8 -*
import os
import torch
from flyai.model.base import Base
from path import MODEL_PATH, DATA_PATH
from torchvision.models.segmentation import *
import numpy as np
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as F
Torch_MODEL_NAME = "model.pkl"

cuda_avail = torch.cuda.is_available()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.mean, self.std = self._compute_mean_std()
        self.T = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])

    def predict(self, **data):
        # cnn =  deeplabv3_resnet101(pretrained=False, num_classes=2)
        cnn =  fcn_resnet101(pretrained=False, num_classes=2)
        cnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME)))
        cnn.to(device)
        cnn.eval()
        x_data = self.data.predict_data(**data)
        print(self.mean, self.std)
        x_data = torch.from_numpy(x_data)
        for i in range(x_data.shape[0]):
            x_data[i] = F.normalize(x_data[i],self.mean,self.std)
        x_data = x_data.float().to(device)
        outputs = cnn(x_data)
        outputs = outputs['out'].max(dim=1)[1]
        outputs = outputs.cpu()
        prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        print(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
        # cnn =  deeplabv3_resnet101(pretrained=False, num_classes=2)
        cnn =  fcn_resnet101(pretrained=False, num_classes=2)
        cnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME)))
        cnn.to(device)
        cnn.eval()
        labels = []
        with torch.no_grad():
            for data in datas:
                x_data = self.data.predict_data(**data)
                x_data = torch.from_numpy(x_data)
                for i in range(x_data.shape[0]):
                    x_data[i] = F.normalize(x_data[i],self.mean,self.std)
                x_data = x_data.float().to(device)
                outputs = cnn(x_data)
                outputs = outputs['out'].max(dim=1)[1]
                outputs = outputs.cpu()
                prediction = outputs.data.numpy()
                prediction = self.data.to_categorys(prediction)
                labels.append(prediction)
        return labels


    def save_model(self, network, path, name=Torch_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network.state_dict(), os.path.join(path, name))

    def _compute_mean_std(self):

        print('Computing mean and std ...')
        x_train  = self.data.get_all_data()[0]

        num_pixel = 0
        s1 = np.zeros(3)
        s2 = np.zeros(3)

        for x in x_train:
            img_path = os.path.join(DATA_PATH, x['image_path'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img2 = img**2
            s1 += img.sum(0).sum(0)
            s2 += img2.sum(0).sum(0)
            num_pixel += img.size

        means = s1 / num_pixel
        stds = np.sqrt(s2 / num_pixel - means**2)
        print(means, stds)

        return means, stds

    def get_mean_std(self):
        return self.mean, self.std
