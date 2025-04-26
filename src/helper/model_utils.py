import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .boundary import boundaries_mirror_y, masks_to_boundary


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
