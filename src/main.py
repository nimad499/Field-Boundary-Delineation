from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from InquirerPy import inquirer
from PIL import Image

import image_crop
import image_download
from helper import (
    get_dataset,
    masks_to_boundary,
    model_class_options,
    models,
    model_masks_output,
)

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
            selected_model_name = inquirer.select(
                message="Select a model: ",
                choices=models.keys(),
                pointer="=>",
            ).execute()
            (
                model,
                appropriate_dataset,
                appropriate_trainer,
            ) = model_class_options(models[selected_model_name])
            selected_model = model()

            dataset = get_dataset(appropriate_dataset)

            output_dir = Path(
                input("Enter output dir(for saving log, model and ...): ")
            )

            num_epochs = int(input("How many epochs do you want to train for? "))

            batch_size = int(input("What batch size do you want to use? "))

            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            params = [p for p in selected_model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                params,
                lr=0.005,
                momentum=0.9,
                weight_decay=0.0005,
            )

            trainer = appropriate_trainer(selected_model, optimizer, output_dir, device)
            trainer.train(
                dataset,
                num_epochs,
                batch_size,
            )

        case _Mode.CONTINUE_TRAINING:
            checkpoint_path = Path(input("Enter the checkpoint output path: "))
            checkpoint = torch.load(
                checkpoint_path / "model" / "best_model.tar",
                weights_only=False,
            )

            model = checkpoint["model"]
            optimizer = checkpoint["optimizer"]
            current_epoch = checkpoint["epoch"]

            _, appropriate_dataset, appropriate_trainer = model_class_options(
                model.__class__
            )

            dataset = get_dataset(appropriate_dataset)

            output_dir = input("Enter output dir(press enter to use current path): ")
            if output_dir == "":
                output_dir = checkpoint_path
            else:
                output_dir = Path(output_dir)

            num_epochs = int(input("How many epochs do you want to train for? "))

            batch_size = int(input("What batch size do you want to use? "))

            trainer = appropriate_trainer(
                model, optimizer, output_dir, current_epoch=current_epoch
            )
            trainer.train(dataset, num_epochs, batch_size)

        case _Mode.INFERENCE:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            checkpoint_path = Path(input("Enter the checkpoint output path: "))
            checkpoint = torch.load(
                checkpoint_path / "model" / "best_model.tar",
                weights_only=False,
            )

            model: torch.nn.Module = checkpoint["model"]
            model.eval()
            model.to(device)

            image_path = Path(input("Enter the image path: "))
            image = Image.open(image_path)
            image = TF.to_tensor(image).unsqueeze(0).to(device)

            output_dir = input("Enter output dir(press enter to use image path): ")
            if output_dir == "":
                output_dir = image_path
            else:
                output_dir = Path(output_dir)

            masks = model_masks_output(model, image)
            boundaries = masks_to_boundary((masks * 255).astype(np.uint8))
            boundaries.to_file(output_dir / f"{image_path.stem}.shp")
