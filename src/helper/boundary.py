import cv2
import geopandas as gpd
from shapely.geometry import Polygon


def masks_to_boundary(masks, threshold=64):
    boundaries = list()
    for mask in masks:
        _, thresh = cv2.threshold(mask, threshold, 255, 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            if contour.shape[0] > 2:
                polygon = Polygon(contour[:, 0, :])
                boundaries.append(polygon)

    boundaries = gpd.GeoDataFrame(geometry=boundaries)

    return boundaries


def boundaries_mirror_y(boundaries: gpd.GeoDataFrame):
    boundaries_copy = boundaries.copy()

    boundaries_copy["geometry"] = boundaries_copy["geometry"].apply(
        lambda g: Polygon([(abs(x), -abs(y)) for x, y in g.exterior.coords])
    )

    return boundaries_copy
