import torch
from torch.utils.data import ConcatDataset
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.segmentation import DeepLabV3

from dataset import (
    InstanceSegmentationLazyDataset,
    SemanticSegmentationLazyDataset,
)
from model import deeplabv3_model, mask_rcnn_model
from train import InstanceSegmentationTrain, SemanticSegmentationTrain

model_name_class = {"Mask-RCNN": MaskRCNN, "DeepLab": DeepLabV3}
model_class_options = {
    MaskRCNN: (
        mask_rcnn_model,
        InstanceSegmentationLazyDataset,
        InstanceSegmentationTrain,
    ),
    DeepLabV3: (
        deeplabv3_model,
        SemanticSegmentationLazyDataset,
        SemanticSegmentationTrain,
    ),
}


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
    ) = model_class_options[model_name_class[model_architecture]]
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
