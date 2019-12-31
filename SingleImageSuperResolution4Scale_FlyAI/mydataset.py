import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from path import DATA_PATH
import os.path as osp
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
class MyData(Dataset):
    def __init__(self, lr_file, hr_file, transformation=None):
        self.lr_file = lr_file
        self.hr_file = hr_file
        self.transformation = transformation

    def __len__(self):
        return len(self.lr_file)

    def __getitem__(self, index):
        lr = self._load_lr(index)
        hr = self._load_hr(index)

        if self.transformation == 'train':
            lr, hr = get_patch(
                lr, hr,
                patch_size=224,
                scale=4,
                multi=False,
                input_large=False
            )
            lr, hr = augment(lr, hr)
        elif self.transformation == 'valid':
            lr, hr = get_patch(
                lr, hr,
                patch_size=224,
                scale=4,
                multi=False,
                input_large=False
            )
            # ih, iw = lr.shape[:2]
            # hr = hr[0:ih * 4, 0:iw * 4]

        lr, hr = np2Tensor(lr, hr)
        # lr = F.to_tensor(lr)
        # hr = F.to_tensor(hr)

        return lr, hr

    def _load_lr(self, index):
        img_path = osp.join(DATA_PATH, self.lr_file[index]['lr_image_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_hr(self, index):
        img_path = osp.join(DATA_PATH, self.hr_file[index]['hr_image_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

# def set_channel(*args, n_channels=3):
#     def _set_channel(img):
#         if img.ndim == 2:
#             img = np.expand_dims(img, axis=2)
#
#         c = img.shape[2]
#         if n_channels == 1 and c == 3:
#             img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
#         elif n_channels == 3 and c == 1:
#             img = np.concatenate([img] * n_channels, 2)
#
#         return img
#
#     return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]
