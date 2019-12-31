import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from path import DATA_PATH
import os.path as osp
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as F

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
class SkyData(Dataset):
    def __init__(self, image_file, label_file, transformation=None):
        self.image_file = image_file
        self.label_file = label_file
        self.transformation = transformation

    def __len__(self):
        return len(self.image_file)

    def __getitem__(self, index):
        img = self._load_image(index)
        mask = self._load_mask(index)

        if self.transformation is not None:
            augmented = self.transformation(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = F.to_tensor(img)
        mask = torch.as_tensor(mask, dtype=torch.int64)

        return img, mask

    def _load_image(self, index):
        img_path = osp.join(DATA_PATH, self.image_file[index]['image_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, index):
        img_path = osp.join(DATA_PATH, self.label_file[index]['label_path'])
        img = cv2.imread(img_path)
        img = img[...,0]
        # img[img > 0] = 1
        return img
