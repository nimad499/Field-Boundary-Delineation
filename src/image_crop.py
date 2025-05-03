import argparse
import os
import pathlib

import rasterio
from rasterio.windows import Window


def crop_image(input_image_path: pathlib.Path, output_path, square_size: pathlib.Path):
    output_path = output_path / (input_image_path.name + "_cropped")
    os.makedirs(output_path, exist_ok=True)

    input_image_suffix = input_image_path.suffix

    with rasterio.open(input_image_path) as src:
        window_width, window_height = (
            square_size,
            square_size,
        )

        image_width = src.width
        image_height = src.height

        for row in range(0, image_height, window_height):
            for col in range(0, image_width, window_width):
                window = Window(
                    col_off=col,
                    row_off=row,
                    width=min(window_width, image_width - col),
                    height=min(window_height, image_height - row),
                )

                data = src.read(window=window)

                with rasterio.open(
                    output_path
                    / f"{int(col / window_width)}_{int(row / window_height)}{input_image_suffix}",
                    "w",
                    height=window.height,
                    width=window.width,
                    count=src.count,
                    dtype=src.dtypes[0],
                ) as dest:
                    dest.write(data)


def main_function():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Input image", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory",
        required=True,
        type=pathlib.Path,
    )
    parser.add_argument("-s", "--size", help="Square size", required=True, type=int)
    args = parser.parse_args()

    input_image_path = args.input
    output_dir = args.output
    size = args.size

    crop_image(input_image_path, output_dir, size)

    print("Image cropped")


if __name__ == "__main__":
    main_function()
