from pathlib import Path


def get_dataset():
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
        boundaries_path = Path(input("Enter correspond boundaries path: "))
        data_paths.append((image_path, boundaries_path))

    return data_paths
