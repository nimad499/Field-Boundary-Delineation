# Field Boundary Delineation
## How to add a new model?
To add a new model, start by writing a function that returns your model in `src/model.py`. For example:
```py
def deeplabv3_model():
    model = deeplabv3_resnet50(weights="DEFAULT")

    in_channels = model.classifier[4].in_channels
    kernel_size = model.classifier[4].kernel_size
    model.classifier[4] = torch.nn.Conv2d(
        in_channels, 2, kernel_size=kernel_size
    )

    return model
```
Next, add the model’s name and class to the `models` dictionary in `src/helper.py`:
```py
models = {"DeepLab": DeepLabV3}
```
Finally, add the model option, along with its associated training and dataset classes, in the `model_class_options` function in `src/helper.py`:
```py
elif issubclass(target_class, DeepLabV3):
    return (
        deeplabv3_model,
        SemanticSegmentationLazyDataset,
        SemanticSegmentationTrain,
    )
```