import importlib
import threading


def preload_modules():
    def import_modules():
        for module in (
            "torch",
            "torchvision.transforms.functional",
            "geopandas",
            "requests",
            "pystac_client",
            "InquirerPy",
        ):
            importlib.import_module(module)

    threading.Thread(target=import_modules, daemon=True).start()
