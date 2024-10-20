from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import Polygon
from torch.utils.data import ConcatDataset
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.segmentation import DeepLabV3

from dataset import (
    InstanceSegmentationLazyDataset,
    SemanticSegmentationLazyDataset,
)
from model import deeplabv3_model, mask_rcnn_model
from train import InstanceSegmentationTrain, SemanticSegmentationTrain

models = {"Mask-RCNN": MaskRCNN, "DeepLab": DeepLabV3}


def model_class_options(target_class):
    if issubclass(target_class, MaskRCNN):
        return (
            mask_rcnn_model,
            InstanceSegmentationLazyDataset,
            InstanceSegmentationTrain,
        )
    elif issubclass(target_class, DeepLabV3):
        return (
            deeplabv3_model,
            SemanticSegmentationLazyDataset,
            SemanticSegmentationTrain,
        )
    else:
        raise ValueError


def get_dataset(appropriate_dataset):
    data_paths = []
    image_path = Path(input("Enter image path: "))
    boundaries_path = Path(input("Enter correspond boundaries path: "))
    data_paths.append((image_path, boundaries_path))
    while True:
        image_path = input(
            "Enter another image path (press enter if you done): "
        ).strip()
        if image_path != "":
            image_path = Path(image_path)
        else:
            break
        boundaries_path = Path(input("Enter correspond boundaries path: "))
        data_paths.append((image_path, boundaries_path))

    dataset = ConcatDataset([appropriate_dataset(i, b) for i, b in data_paths])

    return dataset


def masks_to_boundary(masks, threshold=64):
    boundaries = list()
    for mask in masks:
        _, thresh = cv2.threshold(mask, threshold, 255, 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            if contour.shape[0] > 2:
                polygon = Polygon(contour[:, 0, :])
                boundaries.append(polygon)

    boundaries = gpd.GeoDataFrame(geometry=boundaries)

    return boundaries


def model_masks_output(model, image):
    output = model(image)

    if isinstance(output, list):
        masks = output[0]["masks"].cpu().detach().numpy()[:, 0, :, :]

    elif isinstance(output, dict):
        segmentation_map = (
            torch.argmax(output["out"].squeeze(), dim=0).cpu().detach().numpy()
        )

        masks = segmentation_map[None, :, :]

    else:
        raise ValueError("Unsupported model mask format.")

    return masks
