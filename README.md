# Field Boundary Delineation
This project aims to automate the delineation of agricultural field boundaries using machine learning, specifically through image segmentation algorithms. You can choose between semantic segmentation and instance segmentation models.

Both the input (for training) and output boundaries use the shapefile format, which is a standard in the field of remote sensing. To work with this format, you can use various tools such as QGIS or ArcGIS.

Additionally, satellite images can be easily downloaded from sources such as Planetary Computer, Google Earth Engine, or others. To simplify the download process, a tool has been implemented in this project, allowing you to easily download satellite images from Planetary Computer as a source.
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