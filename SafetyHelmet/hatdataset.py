import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from path import DATA_PATH
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from albumentations import (
    BboxParams,
    HorizontalFlip,
    RandomSizedBBoxSafeCrop,
    RandomCrop,
    Compose
)

class HatData(Dataset):
    def __init__(self, image_file, box_list_dict, transformation=None):
        self.image_file = image_file
        self.box_list_dict = box_list_dict
        self.transforms = transformation

    def __len__(self):
        return len(self.image_file)

    def __getitem__(self, index):
        img = self._load_image(index)
        boxes, labels = self._load_box(index)

        bunch = {'image': img, 'bboxes': boxes, 'labels': labels}

        if self.transforms is not None:
            bunch = self.transforms(**bunch)

        img = F.to_tensor(bunch['image'])
        boxes = torch.as_tensor(bunch['bboxes'], dtype=torch.float32)
        labels = torch.as_tensor(bunch['labels'], dtype=torch.int64)

        image_id = torch.tensor([index])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        return img, target

    def _load_image(self, index):
        img_path = osp.join(DATA_PATH, self.image_file[index]['img_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_box(self, index):
        box_list = self.box_list_dict[index]['box']
        box_list = box_list.split(' ')
        boxes = []
        labels = []
        for i in range(len(box_list)):
            box = box_list[i]
            box = box.split(',')
            boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])]) #pascal_voc format
            labels.append(int(box[4])+1)
        return boxes, labels


BOX_COLOR = (1.0, 0, 0)
TEXT_COLOR = (1.0, 1, 1)

# def visualize_bbox(img, target, color=BOX_COLOR, thickness=2):
#     img = img.copy()
#     for item in range(len(target["labels"])):
#         x_min, y_min, x_max, y_max = target["boxes"][item]
#         x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
#         cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
#         class_name = str(target["labels"][item])
#         ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
#         cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,TEXT_COLOR, lineType=cv2.LINE_AA)
#
#     plt.imshow(img)
#     plt.show()

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area,
                                               min_visibility=min_visibility, label_fields=['labels']))
def get_transform(train):
    if train:
        return get_aug([HorizontalFlip(p=0.5),RandomSizedBBoxSafeCrop(height=256,width=256)])
    else:
        return get_aug([HorizontalFlip(p=0.5)])


if __name__ == "__main__":
    from flyai.dataset import Dataset
    train_aug = get_transform(True)
    dataset = Dataset(epochs=1, batch=4)
    x_train, y_train, x_val, y_val = dataset.get_all_data()
    train_dataset = HatData(x_train, y_train, train_aug)
    img, target = train_dataset[25]
    # print(img, target)
    img = img.numpy().transpose(1,2,0)
    target['labels'] = target['labels'].numpy()
    target['boxes'] = target['boxes'].numpy()
    # print(img.shape, target)
    # visualize_bbox(img, target)
