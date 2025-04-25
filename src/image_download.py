import atexit
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import planetary_computer as pc
import regex as re
import requests
from InquirerPy import inquirer
from pystac_client import Client
from tqdm import tqdm

# ToDo: Add option to download from json file


def _get_valid_date(message):
    while True:
        date = input(message)

        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            print("Invalid format. Please use YYYY-MM-DD.")
            continue

        try:
            datetime.strptime(date, "%Y-%m-%d")
            return date
        except ValueError:
            print("Invalid date. Please check your input.")


def _get_valid_longitude():
    while True:
        lon_str = input("Enter longitude (-180 to 180): ")

        try:
            lon_float = float(lon_str)

            if -180 <= lon_float <= 180:
                return lon_float

            print(f"Longitude must be between -180 and 180. Current value: {lon_float}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def _get_valid_latitude():
    while True:
        lat_str = input("Enter latitude (-90 to 90): ")

        try:
            lat_float = float(lat_str)

            if -90 <= lat_float <= 90:
                return lat_float

            print(f"Latitude must be between -90 and 90. Current value: {lat_float}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def _download_with_progress(url, output_path):
    response = requests.get(url, stream=True, timeout=10)

    total_size = int(response.headers.get("content-length", 0))

    with tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar:
        with open(output_path, "wb") as file:

            def delete_file():
                return os.remove(output_path)

            atexit.register(delete_file)

            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

                progress_bar.update(len(chunk))

            atexit.unregister(delete_file)

    print(f"Download completed. Total size: {total_size} bytes.")


def get_catalog(url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"):
    catalog = Client.open(
        url,
        modifier=pc.sign_inplace,
    )

    return catalog


def catalog_search(
    catalog: Client, start_date: str, end_date: str, lon, lat, collection: str
):
    date_range = {"interval": [start_date, end_date]}
    point_geometry = {"type": "Point", "coordinates": [lon, lat]}

    search_result = catalog.search(
        filter_lang="cql2-json",
        filter={
            "op": "and",
            "args": [
                {
                    "op": "anyinteracts",
                    "args": [{"property": "datetime"}, date_range],
                },
                {
                    "op": "s_intersects",
                    "args": [{"property": "geometry"}, point_geometry],
                },
                {
                    "op": "=",
                    "args": [{"property": "collection"}, collection],
                },
            ],
        },
    )

    return search_result


def main_function():
    catalog = get_catalog()

    fetch_collections = inquirer.select(
        message="Do you want to fetch the collections: ",
        choices=("Yes", "No"),
        pointer="=>",
    ).execute()
    fetch_collections = fetch_collections == "Yes"

    if fetch_collections:
        selected_collection = inquirer.select(
            message="Select a collection: ",
            choices=sorted((c.id for c in catalog.get_collections())),
            pointer="=>",
        ).execute()
    else:
        selected_collection = input("Enter collection id: ").strip()

    lon = _get_valid_longitude()
    lat = _get_valid_latitude()

    while True:
        start_date = _get_valid_date("Enter start date in (YYYY-MM-DD) format: ")
        end_date = _get_valid_date("Enter end date in (YYYY-MM-DD) format: ")

        if (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            > datetime.strptime(end_date, "%Y-%m-%d").date()
        ):
            print("Start date must be before End date.")
            continue

        break

    search_result = catalog_search(
        catalog, start_date, end_date, lon, lat, selected_collection
    )

    items = search_result.item_collection()
    items_df = gpd.GeoDataFrame.from_features(items.to_dict(), crs="epsg:4326")
    _items_df_cols_filter = ["datetime", "eo:cloud_cover"]
    print(items_df[_items_df_cols_filter])
    # ToDo: Check entered number
    selected_item_index = int(input("Select an item: "))
    selected_item = items[selected_item_index]

    selected_format = inquirer.select(
        message="Select an option: ",
        choices=selected_item.assets.keys(),
        pointer="=>",
    ).execute()

    selected_item_id = selected_item.id
    output_path = Path(input("Enter the output path: ")) / selected_item_id
    output_path.mkdir(exist_ok=True, parents=True)

    image_url = selected_item.assets[selected_format].href

    parsed_url = urlparse(image_url)
    path_parts = parsed_url.path.split("/")
    file_name = path_parts[-1]
    file_name_stem = Path(file_name).stem

    _download_with_progress(image_url, output_path / file_name)

    with open(output_path / f"{file_name_stem}.json", "w", encoding="utf-8") as f:
        json.dump(selected_item.to_dict(), f, indent=4, sort_keys=False)

    print(f"Output path: {output_path.absolute()}")
    print(f"Output file: {file_name}")


if __name__ == "__main__":
    main_function()
