from enum import Enum
from pathlib import Path

import torch
from InquirerPy import inquirer

from helper import get_dataset, model_class_options, models


if __name__ == "__main__":

    class _Mode(Enum):
        TRAIN_NEW_MODEL = "Train new model"
        CONTINUE_TRAINING = "Continue training"
        INFERENCE = "Inference"

    mode = selected_format = inquirer.select(
        message="Select a mode: ",
        choices=[m.value for m in _Mode],
        pointer="=>",
    ).execute()
    mode = _Mode(mode)

    match mode:
        case _Mode.TRAIN_NEW_MODEL:
            selected_model_name = selected_format = inquirer.select(
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

            num_epochs = int(
                input("How many epochs do you want to train for? ")
            )

            batch_size = int(input("What batch size do you want to use? "))

            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            params = [
                p for p in selected_model.parameters() if p.requires_grad
            ]
            optimizer = torch.optim.SGD(
                params,
                lr=0.005,
                momentum=0.9,
                weight_decay=0.0005,
            )

            trainer = appropriate_trainer(
                selected_model, optimizer, output_dir, device
            )
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

            output_dir = input(
                "Enter output dir(press enter to use current path): "
            )
            if output_dir == "":
                output_dir = checkpoint_path
            else:
                Path(output_dir)

            num_epochs = int(
                input("How many epochs do you want to train for? ")
            )

            batch_size = int(input("What batch size do you want to use? "))

            trainer = appropriate_trainer(
                model, optimizer, output_dir, current_epoch=current_epoch
            )
            trainer.train(dataset, num_epochs, batch_size)
        case _Mode.INFERENCE:
            raise NotImplementedError
