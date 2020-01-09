from torch.utils.data.dataset import Dataset
from path import DATA_PATH
import os.path as osp
import cv2
import torchvision.transforms.functional as F
import torch


class CinicData(Dataset):
    def __init__(self, image_file, label_file, transformation=None):
        self.image_file = image_file
        self.label_file = label_file
        self.transformation = transformation
        self.n_classes = 10

    def __len__(self):
        return len(self.image_file)

    def __getitem__(self, index):
        img = self._load_image(index)
        label = self._load_label(index)

        if self.transformation is not None:
            augmented = self.transformation(image=img)
            img = augmented['image']

        img = F.to_tensor(img)
        label = torch.tensor(label)
        # label = torch.from_numpy(label)

        return img, label

    def _load_image(self, index):
        img_path = osp.join(DATA_PATH, self.image_file[index]['image_path'])
        img = cv2.imread(img_path)
        if img.shape != (32,32,3): print("img shape: ", img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_label(self, index):
        label = self.label_file[index]['labels']
        return label
