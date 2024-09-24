from model import mask_rcnn_model
from InquirerPy import inquirer
from train import InstanceSegmentationTrain
import torch
from torch.utils.data import ConcatDataset
from pathlib import Path
from dataset import InstanceSegmentationLazyDataset
from enum import Enum

# ToDo: Get config from user

_models = {
    "Mask-RCNN": (
        mask_rcnn_model,
        InstanceSegmentationLazyDataset,
        InstanceSegmentationTrain,
    )
}

if __name__ == "__main__":

    class _Mode(Enum):
        train_nem_model = "Train new model"
        continue_training = "Continue training"
        inference = "Inference"

    mode = selected_format = inquirer.select(
        message="Select a mode: ",
        choices=[m.value for m in _Mode],
        pointer="=>",
    ).execute()
    mode = _Mode(mode)

    match mode:
        case _Mode.train_nem_model:
            selected_model_name = selected_format = inquirer.select(
                message="Select a model: ",
                choices=_models.keys(),
                pointer="=>",
            ).execute()
            model, appropriate_dataset, appropriate_trainer = _models[
                selected_model_name
            ]
            selected_model = model()

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
                boundaries_path = Path(
                    input("Enter correspond boundaries path: ")
                )
                data_paths.append((image_path, boundaries_path))

            dataset = ConcatDataset(
                [appropriate_dataset(i, b) for i, b in data_paths]
            )

            output_dir = Path(
                input("Enter output dir(for saving log, model and ...): ")
            )

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
            trainer.train(dataset, 100, 2)
        case _Mode.continue_training:
            raise NotImplementedError
        case _Mode.inference:
            raise NotImplementedError
