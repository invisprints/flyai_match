# -*- coding: utf-8 -*
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model(num_classes, pretrained=False):
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return net

