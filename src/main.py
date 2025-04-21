from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from InquirerPy import inquirer
from PIL import Image
from torch.utils.data import ConcatDataset

import image_crop
import image_download
from helper import (
    boundaries_mirror_y,
    get_dataset,
    masks_to_boundary,
    model_class_options,
    model_masks_output,
    models,
)


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
    model_path, dataset_paths, output_dir_path, num_epochs, batch_size
):
    checkpoint = torch.load(
        model_path,
        weights_only=False,
    )

    model = checkpoint["model"]
    optimizer = checkpoint["optimizer"]
    current_epoch = checkpoint["epoch"]

    _, appropriate_dataset, appropriate_trainer = model_class_options(model.__class__)
    dataset = ConcatDataset([appropriate_dataset(i, b) for i, b in dataset_paths])

    trainer = appropriate_trainer(
        model, optimizer, output_dir_path, current_epoch=current_epoch
    )
    trainer.train(dataset, num_epochs, batch_size)


def inference(model_path, image_path, output_dir_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    checkpoint = torch.load(
        model_path,
        weights_only=False,
    )

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


if __name__ == "__main__":

    class _Mode(Enum):
        DOWNLOAD_IMAGE = "Download Image"
        CROP_IMAGE = "Crop Image"
        TRAIN_NEW_MODEL = "Train new model"
        CONTINUE_TRAINING = "Continue training"
        INFERENCE = "Inference"

    mode = inquirer.select(
        message="Select a mode: ",
        choices=[m.value for m in _Mode],
        pointer="=>",
    ).execute()
    mode = _Mode(mode)

    match mode:
        case _Mode.DOWNLOAD_IMAGE:
            image_download.main_function()

        case _Mode.CROP_IMAGE:
            input_path = Path(input("Enter input image path: "))
            output_dir = Path(input("Enter output directory: "))
            square_size = int(input("Enter square size: "))

            image_crop.crop_image(input_path, output_dir, square_size)

        case _Mode.TRAIN_NEW_MODEL:
            model_architecture = inquirer.select(
                message="Select a model: ",
                choices=models.keys(),
                pointer="=>",
            ).execute()

            dataset_paths = get_dataset()

            output_dir_path = Path(
                input("Enter output dir(for saving log, model and ...): ")
            )

            num_epochs = int(input("How many epochs do you want to train for? "))

            batch_size = int(input("What batch size do you want to use? "))

            train_new_model(
                model_architecture,
                dataset_paths,
                output_dir_path,
                num_epochs,
                batch_size,
            )

        case _Mode.CONTINUE_TRAINING:
            checkpoint_path = Path(input("Enter the checkpoint output path: "))
            model_path = checkpoint_path / "model" / "best_model.tar"

            dataset_paths = get_dataset()

            output_dir_path = input(
                "Enter output dir(press enter to use current path): "
            )
            if output_dir_path == "":
                output_dir_path = checkpoint_path
            else:
                output_dir_path = Path(output_dir_path)

            num_epochs = int(input("How many epochs do you want to train for? "))

            batch_size = int(input("What batch size do you want to use? "))

            continue_training(
                model_path,
                dataset_paths,
                output_dir_path,
                num_epochs,
                batch_size,
            )

        case _Mode.INFERENCE:
            checkpoint_path = Path(input("Enter the checkpoint output path: "))
            model_path = checkpoint_path / "model" / "best_model.tar"

            image_path = Path(input("Enter the image path: "))

            output_dir_path = Path(input("Enter the output directory: "))

            inference(model_path, image_path, output_dir_path)
