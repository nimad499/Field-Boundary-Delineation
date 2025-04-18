import json
import tkinter
from datetime import datetime
from pathlib import Path
from tkinter import IntVar, StringVar, Tk, Toplevel, filedialog, messagebox, ttk
from urllib.parse import urlparse

import geopandas as gpd
import requests
from tqdm import tqdm

from helper import models
from image_crop import crop_image
from image_download import catalog_search, get_catalog
from main import continue_training, inference, train_new_model


def _download_image(url, output_path):
    response = requests.get(url, stream=True, timeout=10)
    total_size = int(response.headers.get("content-length", 0))
    with (
        tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar,
        open(output_path, "wb") as file,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            progress_bar.update(len(chunk))
    messagebox.showinfo("Download Completed", f"Downloaded file: {output_path}")


def download_image_window():
    new_window = Toplevel(root)
    new_window.title("Download Image")
    new_window.geometry("500x900")

    catalog = get_catalog()

    collection_var = StringVar()
    lon_var = StringVar()
    lat_var = StringVar()
    start_date_var = StringVar()
    end_date_var = StringVar()
    output_path_var = StringVar()
    selected_format_var = StringVar()

    search_results = []
    items_df = None

    def search_catalog():
        nonlocal search_results, items_df
        try:
            lon = float(lon_var.get())
            lat = float(lat_var.get())
            start_date = datetime.strptime(start_date_var.get(), "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_var.get(), "%Y-%m-%d").date()

            if start_date > end_date:
                messagebox.showerror("Error", "Start date must be before End date.")
                return

            search_results = catalog_search(
                catalog, str(start_date), str(end_date), lon, lat, collection_var.get()
            )
            items = search_results.item_collection()
            items_df = gpd.GeoDataFrame.from_features(items.to_dict(), crs="epsg:4326")

            items_listbox.delete(0, "end")

            if items_df.empty:
                messagebox.showinfo(
                    "No Results",
                    "No images found for the given location and date range.",
                )
                return

            for i, row in items_df.iterrows():
                items_listbox.insert(
                    "end",
                    f"{i}: {row['datetime']} | Cloud Cover: {row.get('eo:cloud_cover', 'N/A')}",
                )

        except ValueError:
            messagebox.showerror(
                "Error", "Invalid longitude, latitude, or date format."
            )

    def select_output_directory():
        path = filedialog.askdirectory()
        if path:
            output_path_var.set(path)

    def fetch_formats():
        try:
            selection = items_listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select an image from the list.")
                return
            index = selection[0]
            selected_item = search_results.item_collection()[index]
            formats_list = list(selected_item.assets.keys())

            format_dropdown["values"] = formats_list
            format_dropdown.current(0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch formats: {e}")

    def start_download():
        try:
            selection = items_listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select an image from the list.")
                return
            index = selection[0]
            selected_item = search_results.item_collection()[index]
            selected_format = selected_format_var.get()

            image_url = selected_item.assets[selected_format].href
            output_folder = Path(output_path_var.get())
            output_folder.mkdir(parents=True, exist_ok=True)

            file_name = Path(urlparse(image_url).path).name
            file_path = output_folder / file_name

            _download_image(image_url, file_path)

            with open(output_folder / f"{file_name}.json", "w", encoding="utf-8") as f:
                json.dump(selected_item.to_dict(), f, indent=4)

        except Exception as e:
            messagebox.showerror("Error", f"Download failed: {e}")

    # UI Elements
    ttk.Label(new_window, text="Select Collection:", font=("Arial", 12)).pack(pady=5)
    collection_dropdown = ttk.Combobox(
        new_window,
        textvariable=collection_var,
        values=[c.id for c in catalog.get_all_collections()],
        font=("Arial", 12),
    )
    collection_dropdown.pack(pady=5, padx=10, fill="x", expand=True)
    collection_dropdown.current(0)

    for label, var in [
        ("Longitude (-180 to 180):", lon_var),
        ("Latitude (-90 to 90):", lat_var),
        ("Start Date (YYYY-MM-DD):", start_date_var),
        ("End Date (YYYY-MM-DD):", end_date_var),
    ]:
        ttk.Label(new_window, text=label, font=("Arial", 12)).pack(pady=5)
        ttk.Entry(new_window, textvariable=var, font=("Arial", 10)).pack(
            pady=5, padx=10, fill="x"
        )

    ttk.Button(new_window, text="Search Images", command=search_catalog).pack(pady=10)

    items_listbox = tkinter.Listbox(new_window, height=5)
    items_listbox.pack(pady=5, padx=10, fill="both", expand=True)

    ttk.Button(new_window, text="Fetch Available Formats", command=fetch_formats).pack(
        pady=10
    )

    ttk.Label(new_window, text="Select Image Format:", font=("Arial", 12)).pack(pady=5)
    format_dropdown = ttk.Combobox(new_window, textvariable=selected_format_var)
    format_dropdown.pack(pady=5)

    ttk.Label(new_window, text="Output Directory:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=output_path_var, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_output_directory).pack(pady=5)

    ttk.Button(new_window, text="Download Image", command=start_download).pack(pady=10)


def crop_image_window():
    new_window = Toplevel(root)
    new_window.title("Crop Image")
    new_window.geometry("400x400")

    image_path = StringVar()
    output_path = StringVar()
    crop_size = IntVar()

    def select_image_file():
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.tif;*.tiff"), ("All Files", "*.*")]
        )
        if path:
            image_path.set(path)

    def select_output_directory():
        path = filedialog.askdirectory()
        if path:
            output_path.set(path)

    def run_crop_image():
        if not image_path.get():
            messagebox.showerror("Error", "Please select an image file.")
            return
        if not output_path.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return
        try:
            size = crop_size.get()
            if size <= 0:
                raise ValueError("Crop size must be a positive integer.")
        except ValueError:
            messagebox.showerror(
                "Error", "Invalid crop size. Enter a positive integer."
            )
            return

        try:
            crop_image(image_path.get(), output_path.get(), size)
            messagebox.showinfo("Success", "Image cropped successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to crop image: {e}")

    ttk.Label(new_window, text="Image File:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=image_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_image_file).pack(pady=5)

    ttk.Label(new_window, text="Output Path:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=output_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_output_directory).pack(pady=5)

    ttk.Label(new_window, text="Crop Size (pixels):", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=crop_size, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )

    ttk.Button(new_window, text="Crop", command=run_crop_image).pack(pady=20)


def train_new_model_window():
    new_window = Toplevel(root)
    new_window.title("Train New Model")
    new_window.geometry("400x650")

    images_path = StringVar()
    boundaries_path = StringVar()
    output_path = StringVar()
    num_epochs = IntVar()
    batch_size = IntVar()
    model_architecture = StringVar()

    def select_images_path():
        path = filedialog.askdirectory()
        if path:
            images_path.set(path)

    def select_boundaries_path():
        path = filedialog.askdirectory()
        if path:
            boundaries_path.set(path)

    def select_output_directory():
        path = filedialog.askdirectory()
        if path:
            output_path.set(path)

    def run_train_new_model():
        if not images_path.get():
            messagebox.showerror("Error", "Please select images path.")
            return
        if not boundaries_path.get():
            messagebox.showerror("Error", "Please select boundaries path.")
            return
        if not output_path.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return
        try:
            epochs = num_epochs.get()
            if epochs <= 0:
                raise ValueError("Number of epochs must be a positive integer.")
        except ValueError:
            messagebox.showerror(
                "Error", "Invalid number of epochs. Please enter a positive integer."
            )
            return
        try:
            batch = batch_size.get()
            if batch <= 0:
                raise ValueError("Batch size must be a positive integer.")
        except ValueError:
            messagebox.showerror(
                "Error", "Invalid batch size. Please enter a positive integer."
            )
            return

        train_new_model(
            model_architecture.get(),
            [(Path(images_path.get()), Path(boundaries_path.get()))],
            Path(output_path.get()),
            num_epochs.get(),
            batch_size.get(),
        )

    ttk.Label(new_window, text="Images Path:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=images_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_images_path).pack(pady=5)

    ttk.Label(new_window, text="Boundaries Path:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=boundaries_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_boundaries_path).pack(pady=5)

    ttk.Label(new_window, text="Output Path:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=output_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_output_directory).pack(pady=5)

    ttk.Label(new_window, text="Number of Epochs:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=num_epochs, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )

    ttk.Label(new_window, text="Batch Size:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=batch_size, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )

    ttk.Label(new_window, text="Model Architecture:", font=("Arial", 12)).pack(pady=5)
    for option in models.keys():
        ttk.Radiobutton(
            new_window,
            text=option,
            variable=model_architecture,
            value=option,
        ).pack()

    ttk.Button(new_window, text="Train", command=run_train_new_model).pack(pady=20)


def continue_training_window():
    new_window = Toplevel(root)
    new_window.title("Continue Training")
    new_window.geometry("400x700")

    model_path = StringVar()
    images_path = StringVar()
    boundaries_path = StringVar()
    output_path = StringVar()
    num_epochs = IntVar()
    batch_size = IntVar()

    def select_model_file():
        path = filedialog.askopenfilename(
            filetypes=[("Model File", "*.tar"), ("All Files", "*.*")]
        )
        if path:
            model_path.set(path)

    def select_images_path():
        path = filedialog.askdirectory()
        if path:
            images_path.set(path)

    def select_boundaries_path():
        path = filedialog.askdirectory()
        if path:
            boundaries_path.set(path)

    def select_output_directory():
        path = filedialog.askdirectory()
        if path:
            output_path.set(path)

    def run_continue_training():
        if not model_path.get():
            messagebox.showerror("Error", "Please select a model file.")
            return
        if not images_path.get():
            messagebox.showerror("Error", "Please select images path.")
            return
        if not boundaries_path.get():
            messagebox.showerror("Error", "Please select boundaries path.")
            return
        if not output_path.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return
        try:
            epochs = num_epochs.get()
            if epochs <= 0:
                raise ValueError("Number of epochs must be a positive integer.")
        except ValueError:
            messagebox.showerror(
                "Error", "Invalid number of epochs. Please enter a positive integer."
            )
            return
        try:
            batch = batch_size.get()
            if batch <= 0:
                raise ValueError("Batch size must be a positive integer.")
        except ValueError:
            messagebox.showerror(
                "Error", "Invalid batch size. Please enter a positive integer."
            )
            return

        continue_training(
            Path(model_path.get()),
            [(Path(images_path.get()), Path(boundaries_path.get()))],
            Path(output_path.get()),
            num_epochs.get(),
            batch_size.get(),
        )

    ttk.Label(new_window, text="Model File:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=model_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_model_file).pack(pady=5)

    ttk.Label(new_window, text="Images Path:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=images_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_images_path).pack(pady=5)

    ttk.Label(new_window, text="Boundaries Path:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=boundaries_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_boundaries_path).pack(pady=5)

    ttk.Label(new_window, text="Output Path:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=output_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_output_directory).pack(pady=5)

    ttk.Label(new_window, text="Number of Epochs:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=num_epochs, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )

    ttk.Label(new_window, text="Batch Size:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=batch_size, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )

    ttk.Button(
        new_window, text="Continue Training", command=run_continue_training
    ).pack(pady=20)


def inference_window():
    new_window = Toplevel(root)
    new_window.title("Inference")
    new_window.geometry("400x500")

    model_path = StringVar()
    image_path = StringVar()
    output_path = StringVar()

    def select_model_file():
        path = filedialog.askopenfilename(
            filetypes=[("Model File", "*.tar"), ("All Files", "*.*")]
        )
        if path:
            model_path.set(path)

    def select_image_file():
        path = filedialog.askopenfilename(
            filetypes=[
                ("Image File", "*.tif"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            image_path.set(path)

    def select_output_directory():
        path = filedialog.askdirectory()
        if path:
            output_path.set(path)

    def run_inference():
        if not model_path.get():
            messagebox.showerror("Error", "Please select a model file.")
            return
        if not image_path.get():
            messagebox.showerror("Error", "Please select an image file.")
            return
        if not output_path.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return

        inference(
            Path(model_path.get()), Path(image_path.get()), Path(output_path.get())
        )

    ttk.Label(new_window, text="Model File:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=model_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_model_file).pack(pady=5)

    ttk.Label(new_window, text="Image File:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=image_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_image_file).pack(pady=5)

    ttk.Label(new_window, text="Output Path:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=output_path, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_output_directory).pack(pady=5)

    ttk.Button(new_window, text="Run Inference", command=run_inference).pack(pady=20)


def open_new_window(title):
    new_window = Toplevel(root)
    new_window.title(title)
    new_window.geometry("400x200")
    ttk.Label(
        new_window, text=f"Welcome to the {title} window!", font=("Arial", 14)
    ).pack(pady=20)


def quit_application():
    root.destroy()


if __name__ == "__main__":
    root = Tk()
    root.title("Field Boundary Delineation")
    root.geometry("400x380")

    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12), padding=10)

    frm = ttk.Frame(root, padding=20)
    frm.pack(fill="both", expand=True)

    ttk.Button(frm, text="Download Image", command=download_image_window).pack(
        pady=5, fill="x"
    )
    ttk.Button(frm, text="Crop Image", command=crop_image_window).pack(pady=5, fill="x")
    ttk.Button(frm, text="Train New Model", command=train_new_model_window).pack(
        pady=5, fill="x"
    )
    ttk.Button(
        frm,
        text="Continue Training",
        command=continue_training_window,
    ).pack(pady=5, fill="x")
    ttk.Button(frm, text="Inference", command=inference_window).pack(pady=5, fill="x")
    ttk.Button(frm, text="Quit", command=quit_application).pack(pady=5, fill="x")

    root.mainloop()
