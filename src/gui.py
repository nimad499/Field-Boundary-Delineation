import json
import os
import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import IntVar, StringVar, Toplevel, filedialog, messagebox
from urllib.parse import urlparse

import ttkbootstrap as ttk

from helper.preload import preload_modules

if getattr(sys, "frozen", False):
    _base_dir = os.path.join(sys._MEIPASS)
    os.environ["GDAL_DATA"] = os.path.join(sys._MEIPASS, "gdal_data")
else:
    _base_dir = os.path.dirname(os.path.abspath(__file__))


def _run_in_thread(function: callable):
    if (
        function in _run_in_thread.active_thread_function.keys()
        and _run_in_thread.active_thread_function[function].is_alive()
    ):
        return

    thread = threading.Thread(target=function, daemon=True)
    thread.start()
    _run_in_thread.active_thread_function[function] = thread


_run_in_thread.active_thread_function = {}


def download_image_window(root_window):
    from PIL import Image, ImageTk

    from image_download import get_catalog

    new_window = Toplevel(root_window)
    new_window.title("Download Image")
    new_window.geometry("500x850")

    collection_var = StringVar()
    lon_var = StringVar()
    lat_var = StringVar()
    start_date_var = StringVar()
    end_date_var = StringVar()
    output_path_var = StringVar()
    selected_format_var = StringVar()

    search_results = None
    cancel_download_event = threading.Event()

    def search_catalog():
        import geopandas as gpd

        from image_download import catalog_search

        nonlocal search_results

        try:
            lon = float(lon_var.get())
            lat = float(lat_var.get())
            start_date = datetime.strptime(start_date_var.get(), "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_var.get(), "%Y-%m-%d").date()

            if start_date > end_date:
                messagebox.showerror("Error", "Start date must be before End date.")
                return

            search_results = catalog_search(
                get_catalog(),
                str(start_date),
                str(end_date),
                lon,
                lat,
                collection_var.get(),
            )
            items = search_results.item_collection()
            items_df = gpd.GeoDataFrame.from_features(items.to_dict(), crs="epsg:4326")

            if items_df.empty:
                messagebox.showinfo(
                    "No Results",
                    "No images found for the given location and date range.",
                )
                return

            items_combobox["values"] = list(
                map(
                    lambda row: f"{row[1]['datetime']} | Cloud Cover: {row[1].get('eo:cloud_cover', 'N/A')}",
                    items_df.iterrows(),
                )
            )
            items_combobox.current(0)

            fetch_formats()

        except ValueError:
            messagebox.showerror(
                "Error", "Invalid longitude, latitude, or date format."
            )

    def select_output_directory():
        path = filedialog.askdirectory()
        if path:
            output_path_var.set(path)

    def fetch_formats():
        index = items_combobox.current()
        selected_item = search_results.item_collection()[index]
        formats_list = list(selected_item.assets.keys())

        format_dropdown["values"] = formats_list
        format_dropdown.current(0)

    class CancelledException(Exception): ...

    def download_image(url, output_path):
        import requests

        response = requests.get(url, stream=True, timeout=10)

        total_size = int(response.headers.get("content-length", 0))

        progress_bar.config(maximum=total_size, value=0)
        progress_text.configure(text=f"0 / {round(total_size / 1048576, 1)} MiB")

        cancel_download_event.clear()

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if cancel_download_event.is_set():
                    raise CancelledException("Download cancelled By User")

                file.write(chunk)

                progress_bar.step(len(chunk))
                progress_text.configure(
                    text=f"{round(progress_bar['value'] / 1048576, 1)} / {round(total_size / 1048576, 1)} MiB"
                )

    def start_download():
        progress_text.configure(text="Starting Download...")

        index = items_combobox.current()
        selected_item = search_results.item_collection()[index]
        selected_format = selected_format_var.get()

        image_url = selected_item.assets[selected_format].href
        output_folder = Path(output_path_var.get())
        output_folder.mkdir(parents=True, exist_ok=True)

        file_name = Path(urlparse(image_url).path).name
        file_path = output_folder / file_name

        try:
            download_image(image_url, file_path)

            with open(output_folder / f"{file_name}.json", "w", encoding="utf-8") as f:
                json.dump(selected_item.to_dict(), f, indent=4)

            messagebox.showinfo("Download Completed", f"Downloaded file: {file_path}")

        except CancelledException:
            if progress_bar.winfo_exists():
                progress_text.configure(text="Download cancelled by user")
            os.remove(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Download failed: {e}")

    def fetch_collections():
        try:
            collections = sorted((c.id for c in get_catalog().get_collections()))
            collection_dropdown["values"] = collections
            if collections:
                collection_dropdown.current(0)
                collection_var.set(collections[0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch collections: {e}")

    def on_dropdown_change(event):
        selected = collection_dropdown.get()
        collection_var.set(selected)

    ttk.Label(new_window, text="Collection Name:", font=("Arial", 12)).pack(pady=5)
    collection_entry = ttk.Entry(
        new_window, textvariable=collection_var, font=("Arial", 10)
    )
    collection_entry.pack(pady=5, padx=10, fill="x")

    dropdown_frame = ttk.Frame(new_window)
    dropdown_frame.pack(pady=5, padx=10, fill="x")

    collection_dropdown = ttk.Combobox(
        dropdown_frame, font=("Arial", 12), state="readonly"
    )
    collection_dropdown.pack(side="left", fill="x", expand=True)
    collection_dropdown.bind("<<ComboboxSelected>>", on_dropdown_change)

    raw_icon = Image.open(os.path.join(_base_dir, "../icon/reload_icon.png"))
    resized_icon = raw_icon.resize((20, 20), Image.Resampling.LANCZOS)
    reload_icon = ImageTk.PhotoImage(resized_icon)
    reload_button = ttk.Button(
        dropdown_frame,
        image=reload_icon,
        command=lambda: _run_in_thread(fetch_collections),
        width=30,
    )
    reload_button.image = reload_icon
    reload_button.pack(side="right", padx=5)

    for label, var in [
        ("Longitude (-180 to 180):", lon_var),
        ("Latitude (-90 to 90):", lat_var),
    ]:
        ttk.Label(new_window, text=label, font=("Arial", 12)).pack(pady=5)
        ttk.Entry(new_window, textvariable=var, font=("Arial", 10)).pack(
            pady=5, padx=10, fill="x"
        )

    for label, var in [
        ("Start Date (YYYY-MM-DD):", start_date_var),
        ("End Date (YYYY-MM-DD):", end_date_var),
    ]:
        ttk.Label(new_window, text=label, font=("Arial", 12)).pack(pady=5)

        date_entry = ttk.DateEntry(new_window, dateformat="%Y-%m-%d")
        date_entry.entry.configure(textvariable=var)
        date_entry.pack(pady=5, padx=10, fill="x")

    ttk.Button(
        new_window, text="Search Images", command=lambda: _run_in_thread(search_catalog)
    ).pack(pady=10)

    items_combobox = ttk.Combobox(new_window, font=("Arial", 12), state="readonly")
    items_combobox.pack(padx=10, pady=5, fill="x", expand=True)
    items_combobox.bind("<<ComboboxSelected>>", lambda _: fetch_formats())

    ttk.Label(new_window, text="Select Image Format:", font=("Arial", 12)).pack(pady=5)
    format_dropdown = ttk.Combobox(
        new_window, textvariable=selected_format_var, state="readonly"
    )
    format_dropdown.pack(pady=5)

    ttk.Label(new_window, text="Output Directory:", font=("Arial", 12)).pack(pady=5)
    ttk.Entry(new_window, textvariable=output_path_var, font=("Arial", 10)).pack(
        pady=5, padx=10, fill="x"
    )
    ttk.Button(new_window, text="Browse", command=select_output_directory).pack(pady=5)

    progress_text = ttk.Label(
        new_window, text=" Download Progress:", font=("Arial", 12)
    )
    progress_text.pack(pady=5)
    progress_bar = ttk.Progressbar(new_window, mode="determinate")
    progress_bar.pack(pady=5, padx=10, fill="x")

    button_frame = ttk.Frame(new_window)
    button_frame.pack(pady=5)
    ttk.Button(
        button_frame,
        text="Download Image",
        command=lambda: _run_in_thread(start_download),
        bootstyle="success",
    ).grid(row=0, column=0, padx=10)
    ttk.Button(
        button_frame,
        text="Cancel",
        command=cancel_download_event.set,
        bootstyle="danger",
    ).grid(row=0, column=1, padx=10)

    new_window.protocol(
        "WM_DELETE_WINDOW", lambda: (cancel_download_event.set(), new_window.destroy())
    )


def crop_image_window(root_window):
    new_window = Toplevel(root_window)
    new_window.title("Crop Image")
    new_window.geometry("400x410")

    image_path = StringVar()
    output_path = StringVar()
    crop_size = IntVar()

    def select_image_file():
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", ["*.tif", "*.tiff"]), ("All Files", "*.*")]
        )
        if path:
            image_path.set(path)

    def select_output_directory():
        path = filedialog.askdirectory()
        if path:
            output_path.set(path)

    def run_crop_image():
        from image_crop import crop_image

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
            crop_image(Path(image_path.get()), Path(output_path.get()), size)
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

    ttk.Button(
        new_window,
        text="Crop",
        command=lambda: _run_in_thread(run_crop_image),
        bootstyle="success",
    ).pack(pady=20)


def train_new_model_window(root_window):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    from helper.training import model_name_class, train_new_model

    new_window = Toplevel(root_window)
    new_window.title("Train New Model")
    new_window.geometry("400x1000")

    images_path = StringVar()
    boundaries_path = StringVar()
    output_path = StringVar()
    num_epochs = IntVar()
    batch_size = IntVar()
    model_architecture = StringVar()
    log_queue = queue.Queue()

    stop_training = threading.Event()

    def on_close():
        stop_training.set()

        plt.close("all")

        new_window.destroy()

    def select_dir(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def cancel_training():
        stop_training.set()
        log_queue.put("[INFO] Training cancelled by user.\n")

    def log_writer():
        try:
            while True:
                msg = log_queue.get_nowait()
                log_text.insert("end", msg)
                log_text.see("end")
        except queue.Empty:
            pass

        new_window.after(100, log_writer)

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

        loss_history.clear()
        stop_training.clear()

        log_queue.put("[INFO] Training started.\n")

        try:
            train_new_model(
                model_architecture.get(),
                [(Path(images_path.get()), Path(boundaries_path.get()))],
                Path(output_path.get()),
                num_epochs.get(),
                batch_size.get(),
                log_queue,
                loss_history,
                stop_training,
            )

            if not stop_training.is_set():
                log_queue.put("[INFO] Training completed.\n")
        except Exception as e:
            log_queue.put(f"[ERROR] {str(e)}\n")

    for label, var, command in [
        ("Images Path:", images_path, lambda: select_dir(images_path)),
        ("Boundaries Path:", boundaries_path, lambda: select_dir(boundaries_path)),
        ("Output Path:", output_path, lambda: select_dir(output_path)),
    ]:
        ttk.Label(new_window, text=label).pack(pady=5)
        ttk.Entry(new_window, textvariable=var).pack(pady=2, fill="x", padx=10)
        ttk.Button(new_window, text="Browse", padding=(2, 1), command=command).pack(
            pady=3
        )

    ttk.Label(new_window, text="Number of Epochs:").pack(pady=5)
    ttk.Entry(new_window, textvariable=num_epochs).pack(pady=2, fill="x", padx=10)

    ttk.Label(new_window, text="Batch Size:").pack(pady=5)
    ttk.Entry(new_window, textvariable=batch_size).pack(pady=2, fill="x", padx=10)

    ttk.Label(new_window, text="Model Architecture:").pack(pady=5)
    for option in model_name_class.keys():
        ttk.Radiobutton(
            new_window, text=option, variable=model_architecture, value=option
        ).pack(anchor="w", padx=20)

    ttk.Label(new_window, text="Training Logs:").pack(pady=5)
    log_text = tk.Text(new_window, height=5)
    log_text.pack(padx=10, pady=2, fill="both", expand=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    loss_history = []

    def update_plot():
        if loss_history:
            ax.clear()
            ax.plot(loss_history, marker="o")
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            canvas.draw()

        new_window.after(1000, update_plot)

    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.get_tk_widget().pack(pady=5, fill="both", expand=True)

    button_frame = ttk.Frame(new_window)
    button_frame.pack(pady=5)
    ttk.Button(
        button_frame,
        text="Start Training",
        command=lambda: _run_in_thread(run_train_new_model),
        bootstyle="success",
    ).grid(row=0, column=0, padx=10)
    ttk.Button(
        button_frame, text="Cancel", command=cancel_training, bootstyle="danger"
    ).grid(row=0, column=1, padx=10)

    log_writer()
    update_plot()

    new_window.protocol("WM_DELETE_WINDOW", on_close)


def continue_training_window(root_window):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    from helper.training import continue_training

    new_window = Toplevel(root_window)
    new_window.title("Continue Training")
    new_window.geometry("400x1000")

    model_path = StringVar()
    images_path = StringVar()
    boundaries_path = StringVar()
    output_path = StringVar()
    num_epochs = IntVar()
    batch_size = IntVar()
    log_queue = queue.Queue()
    loss_history = []

    stop_training = threading.Event()

    def on_close():
        stop_training.set()

        plt.close("all")
        new_window.destroy()

    def select_file(var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def select_dir(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def cancel_training():
        stop_training.set()
        log_queue.put("[INFO] Training cancelled by user.\n")

    def log_writer():
        try:
            while True:
                msg = log_queue.get_nowait()
                log_text.insert("end", msg)
                log_text.see("end")
        except queue.Empty:
            pass
        new_window.after(100, log_writer)

    def update_plot():
        if loss_history:
            ax.clear()
            ax.plot(loss_history, marker="o")
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            canvas.draw()
        new_window.after(1000, update_plot)

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
                raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Invalid number of epochs.")
            return
        try:
            batch = batch_size.get()
            if batch <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Invalid batch size.")
            return

        loss_history.clear()
        stop_training.clear()

        log_queue.put("[INFO] Training started.\n")

        try:
            continue_training(
                Path(model_path.get()),
                [(Path(images_path.get()), Path(boundaries_path.get()))],
                Path(output_path.get()),
                num_epochs.get(),
                batch_size.get(),
                log_queue=log_queue,
                loss_callback_list=loss_history,
                cancel_event=stop_training,
            )

            if not stop_training.is_set():
                log_queue.put("[INFO] Training completed.\n")
        except Exception as e:
            log_queue.put(f"[ERROR] {str(e)}\n")

    for label, var, command in [
        (
            "Model File:",
            model_path,
            lambda: select_file(model_path, [("Model File", "*.tar")]),
        ),
        ("Images Path:", images_path, lambda: select_dir(images_path)),
        ("Boundaries Path:", boundaries_path, lambda: select_dir(boundaries_path)),
        ("Output Path:", output_path, lambda: select_dir(output_path)),
    ]:
        ttk.Label(new_window, text=label).pack(pady=5)
        ttk.Entry(new_window, textvariable=var).pack(pady=2, fill="x", padx=10)
        ttk.Button(new_window, text="Browse", padding=(2, 1), command=command).pack(
            pady=3
        )

    ttk.Label(new_window, text="Number of Epochs:").pack(pady=5)
    ttk.Entry(new_window, textvariable=num_epochs).pack(pady=2, fill="x", padx=10)

    ttk.Label(new_window, text="Batch Size:").pack(pady=5)
    ttk.Entry(new_window, textvariable=batch_size).pack(pady=2, fill="x", padx=10)

    ttk.Label(new_window, text="Training Logs:").pack(pady=5)
    log_text = tk.Text(new_window, height=5)
    log_text.pack(padx=10, pady=5, fill="both", expand=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)

    button_frame = ttk.Frame(new_window)
    button_frame.pack(pady=10)
    ttk.Button(
        button_frame,
        text="Continue Training",
        command=lambda: _run_in_thread(run_continue_training),
        bootstyle="success",
    ).grid(row=0, column=0, padx=10)
    ttk.Button(
        button_frame, text="Cancel", command=cancel_training, bootstyle="danger"
    ).grid(row=0, column=1, padx=10)

    log_writer()
    update_plot()

    new_window.protocol("WM_DELETE_WINDOW", on_close)


def inference_window(root_window):
    new_window = Toplevel(root_window)
    new_window.title("Inference")
    new_window.geometry("400x420")

    model_path = StringVar()
    image_path = StringVar()
    output_path = StringVar()

    def select_file(var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def select_dir(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def run_inference():
        from helper.model_utils import inference

        if not model_path.get():
            messagebox.showerror("Error", "Please select a model file.")
            return
        if not image_path.get():
            messagebox.showerror("Error", "Please select an image file.")
            return
        if not output_path.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return

        run_button.config(state="disabled")

        try:
            inference(
                Path(model_path.get()),
                Path(image_path.get()),
                Path(output_path.get()),
            )
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))
        finally:
            run_button.config(state="normal")

    for label, var, command in [
        (
            "Model File:",
            model_path,
            lambda: select_file(model_path, [("Model File", "*.tar")]),
        ),
        (
            "Image File:",
            image_path,
            lambda: select_file(
                image_path, [("Image File", "*.tif"), ("All Files", "*.*")]
            ),
        ),
        ("Output Path:", output_path, lambda: select_dir(output_path)),
    ]:
        ttk.Label(new_window, text=label, font=("Arial", 12)).pack(pady=5)
        ttk.Entry(new_window, textvariable=var, font=("Arial", 10)).pack(
            pady=2, padx=10, fill="x"
        )
        ttk.Button(new_window, text="Browse", command=command).pack(pady=2)

    run_button = ttk.Button(
        new_window,
        text="Run Inference",
        command=lambda: _run_in_thread(run_inference),
        bootstyle="success",
    )
    run_button.pack(pady=20)

    new_window.protocol("WM_DELETE_WINDOW", new_window.destroy)


def quit_application(root_window):
    root_window.destroy()


if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")
    root.title("Field Boundary Delineation")
    root.geometry("400x360")

    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12), padding=10)

    frm = ttk.Frame(root, padding=20)
    frm.pack(fill="both", expand=True)

    ttk.Button(
        frm, text="Download Image", command=lambda: download_image_window(root)
    ).pack(pady=5, fill="x")
    ttk.Button(frm, text="Crop Image", command=lambda: crop_image_window(root)).pack(
        pady=5, fill="x"
    )
    ttk.Button(
        frm, text="Train New Model", command=lambda: train_new_model_window(root)
    ).pack(pady=5, fill="x")
    ttk.Button(
        frm,
        text="Continue Training",
        command=lambda: continue_training_window(root),
    ).pack(pady=5, fill="x")
    ttk.Button(frm, text="Inference", command=lambda: inference_window(root)).pack(
        pady=5, fill="x"
    )
    ttk.Button(frm, text="Quit", command=lambda: quit_application(root)).pack(
        pady=5, fill="x"
    )

    preload_modules()

    root.mainloop()
