from pathlib import Path

from torch.utils.data import ConcatDataset
from torchvision.models.detection.mask_rcnn import MaskRCNN

from dataset import InstanceSegmentationLazyDataset
from model import mask_rcnn_model
from train import InstanceSegmentationTrain

models = {"Mask-RCNN": MaskRCNN}


def model_class_options(target_class):
    if issubclass(target_class, MaskRCNN):
        return (
            mask_rcnn_model,
            InstanceSegmentationLazyDataset,
            InstanceSegmentationTrain,
        )


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
