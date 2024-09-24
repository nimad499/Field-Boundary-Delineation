import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as F_vis

# ToDo: Transformation. Especially for negative points in shapefiles.


def _files_in_dir(path: Path) -> list[Path]:
    return [
        path / name for name in os.listdir(path) if os.path.isfile(path / name)
    ]


def _get_image_boundary_pair(images_path: Path, boundaries_path: Path):
    images = _files_in_dir(images_path)
    _image_suffixes = [".tif"]
    images = [i for i in images if i.suffix in _image_suffixes]

    boundaries = _files_in_dir(boundaries_path)
    _shapefile_suffix = ".shp"
    boundaries = [b for b in boundaries if b.suffix == _shapefile_suffix]

    pair = []

    # ToDo: Improve this search
    for i in images:
        for b in boundaries:
            if i.stem == b.stem:
                pair.append((i, b))

    return pair


def _polygon_to_mask(polygon, image_shape):
    mask = Image.new("L", image_shape, 0)

    draw = ImageDraw.Draw(mask)

    x, y = polygon.exterior.coords.xy

    # ToDo: Apply this as transformation
    y = -np.array(y)

    polygon_coords = list(zip(x, y))
    draw.polygon(polygon_coords, outline=1, fill=1)

    mask_array = np.array(mask)

    return mask_array


def _polygon_to_box(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    # ToDo: Apply this as transformation
    miny, maxy = -maxy, -miny
    return [minx, miny, maxx, maxy]


def _rasterize_shapefile(shapefile, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)

    for geometry in shapefile.geometry:
        rasterized = _polygon_to_mask(geometry, image_shape)
        mask = np.maximum(mask, rasterized.astype(np.uint8))

    return Image.fromarray(mask)


class InstanceSegmentationLazyDataset(Dataset):
    def __init__(self, images_path: Path, boundaries_path: Path):
        self.image_boundary_pair = _get_image_boundary_pair(
            images_path, boundaries_path
        )

    def __len__(self) -> int:
        return len(self.image_boundary_pair)

    def __getitem__(self, idx: int):
        image_path, boundary_path = self.image_boundary_pair[idx]

        image = np.array(Image.open(image_path))
        boundaries = gpd.read_file(boundary_path)

        masks = []
        boxes = []

        image_shape = image.shape[:2]
        for _, row in boundaries.iterrows():
            polygon = row.geometry
            if polygon is None:
                continue

            mask = _polygon_to_mask(polygon, image_shape)
            masks.append(mask)

            box = _polygon_to_box(polygon)
            boxes.append(box)

        masks = torch.tensor(np.stack(masks))
        boxes = torch.tensor(boxes, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return F_vis.to_tensor(image), target


class SemanticSegmentationLazyDataset(Dataset):
    def __init__(self, images_path: Path, boundaries_path: Path):
        self.image_boundary_pair = _get_image_boundary_pair(
            images_path, boundaries_path
        )

    def __len__(self) -> int:
        return len(self.image_boundary_pair)

    def __getitem__(self, idx: int):
        image_path, boundary_path = self.image_boundary_pair[idx]

        image = np.array(Image.open(image_path))
        boundaries = gpd.read_file(boundary_path)

        image_shape = image.shape[:2]

        mask = _rasterize_shapefile(boundaries, image_shape)

        return F_vis.to_tensor(image), mask
