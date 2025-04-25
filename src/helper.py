from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
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


def get_dataset():
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

    return data_paths


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


def boundaries_mirror_y(boundaries: gpd.GeoDataFrame):
    boundaries_copy = boundaries.copy()

    boundaries_copy["geometry"] = boundaries_copy["geometry"].apply(
        lambda g: Polygon([(abs(x), -abs(y)) for x, y in g.exterior.coords])
    )

    return boundaries_copy


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


def train_new_model(
    model_architecture,
    dataset_paths,
    output_dir_path,
    num_epochs,
    batch_size,
    log_queue=None,
    loss_callback_list=None,
    cancel_callback=None,
):
    (
        model,
        appropriate_dataset,
        appropriate_trainer,
    ) = model_class_options(models[model_architecture])
    selected_model = model()

    dataset = ConcatDataset([appropriate_dataset(i, b) for i, b in dataset_paths])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    params = [p for p in selected_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    trainer = appropriate_trainer(
        selected_model,
        optimizer,
        output_dir_path,
        device,
        log_queue=log_queue,
        loss_callback_list=loss_callback_list,
        cancel_callback=cancel_callback,
    )
    trainer.train(dataset, num_epochs, batch_size)


def continue_training(
    model_path,
    dataset_paths,
    output_dir_path,
    num_epochs,
    batch_size,
    log_queue=None,
    loss_callback_list=None,
    cancel_callback=None,
):
    checkpoint = torch.load(model_path, weights_only=False)

    model = checkpoint["model"]
    optimizer = checkpoint["optimizer"]
    current_epoch = checkpoint["epoch"]

    _, appropriate_dataset, appropriate_trainer = model_class_options(model.__class__)
    dataset = ConcatDataset([appropriate_dataset(i, b) for i, b in dataset_paths])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    trainer = appropriate_trainer(
        model,
        optimizer,
        output_dir_path,
        device,
        current_epoch=current_epoch,
        log_queue=log_queue,
        loss_callback_list=loss_callback_list,
        cancel_callback=cancel_callback,
    )
    trainer.train(dataset, num_epochs, batch_size)


def inference(model_path, image_path, output_dir_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    checkpoint = torch.load(model_path, weights_only=False)
    model: torch.nn.Module = checkpoint["model"]
    model.eval()
    model.to(device)

    image = Image.open(image_path)
    image_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    masks = model_masks_output(model, image_tensor)
    boundaries = masks_to_boundary((masks * 255).astype(np.uint8))
    boundaries_mirrored = boundaries_mirror_y(boundaries)
    boundaries_mirrored.to_file(output_dir_path / f"{image_path.stem}.shp")

    _, ax = plt.subplots()
    ax.imshow(image)
    boundaries.boundary.plot(ax=ax, edgecolor="red")
    plt.show()
