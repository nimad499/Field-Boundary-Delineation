from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules("rasterio")
datas = collect_data_files("rasterio")
