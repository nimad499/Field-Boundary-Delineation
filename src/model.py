import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.segmentation import deeplabv3_resnet50

# ToDo: Add option to set the number of channels


def mask_rcnn_model():
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, dim_reduced, 2
    )

    return model


def deeplabv3_model():
    model = deeplabv3_resnet50(weights="DEFAULT")

    in_channels = model.classifier[4].in_channels
    kernel_size = model.classifier[4].kernel_size
    model.classifier[4] = torch.nn.Conv2d(
        in_channels, 2, kernel_size=kernel_size
    )

    return model
