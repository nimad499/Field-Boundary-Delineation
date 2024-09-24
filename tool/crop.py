import argparse
import os
import pathlib

from PIL import Image


def crop_image(input_image_path, output_path, square_size):
    output_path = output_path / input_image_path.name

    os.makedirs(output_path, exist_ok=True)

    input_image_suffix = input_image_path.suffix

    # ToDo: Slide a window instead of open it at once
    with Image.open(input_image_path) as img:
        width, height = img.size

        cols = int(height / square_size)
        rows = int(width / square_size)

        for row in range(rows):
            for col in range(cols):
                left = col * square_size
                upper = row * square_size
                right = (col + 1) * square_size
                lower = (row + 1) * square_size

                right = min(right, width)
                lower = min(lower, height)

                cropped_img = img.crop((left, upper, right, lower))

                cropped_img.save(
                    output_path / f"{col}_{row}{input_image_suffix}"
                )


if __name__ == "__main__":
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
    parser.add_argument(
        "-s", "--size", help="Square size", required=True, type=int
    )
    args = parser.parse_args()

    input_image_path = args.input
    output_dir = args.output
    size = args.size

    crop_image(input_image_path, output_dir, size)

    print("Image cropped")
