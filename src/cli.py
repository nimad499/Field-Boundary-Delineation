from enum import Enum
from pathlib import Path

from InquirerPy import inquirer

import image_crop
import image_download
from helper import (
    continue_training,
    get_dataset,
    inference,
    model_name_class,
    train_new_model,
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
            model_architecture = inquirer.select(
                message="Select a model: ",
                choices=model_name_class.keys(),
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
