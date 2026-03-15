"""
Auto Censor App - GUI tool for censoring images and videos.
Supports GPU acceleration via NVIDIA CUDA.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

import cv2

# Suppress CUDA DLL load errors on systems without CUDA installed
os.environ.setdefault("ORT_CUDA_UNAVAILABLE_OK", "1")
try:
    import onnxruntime
except OSError:
    # CUDA DLLs missing — force CPU-only mode
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import onnxruntime

from nudenet import NudeDetector

# Use bundled ffmpeg if available
APP_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
_ffmpeg_local = os.path.join(APP_DIR, "ffmpeg.exe")
_ffprobe_local = os.path.join(APP_DIR, "ffprobe.exe")
FFMPEG = _ffmpeg_local if os.path.exists(_ffmpeg_local) else "ffmpeg"
FFPROBE = _ffprobe_local if os.path.exists(_ffprobe_local) else "ffprobe"


# --- Detection classes ---
ALL_CLASSES = {
    "Face (Female)": "FACE_FEMALE",
    "Face (Male)": "FACE_MALE",
    "Breast (Exposed)": "FEMALE_BREAST_EXPOSED",
    "Breast (Covered)": "FEMALE_BREAST_COVERED",
    "Female Genitalia (Exposed)": "FEMALE_GENITALIA_EXPOSED",
    "Female Genitalia (Covered)": "FEMALE_GENITALIA_COVERED",
    "Male Genitalia (Exposed)": "MALE_GENITALIA_EXPOSED",
    "Male Genitalia (Covered)": "MALE_GENITALIA_COVERED",
    "Buttocks (Exposed)": "BUTTOCKS_EXPOSED",
    "Buttocks (Covered)": "BUTTOCKS_COVERED",
    "Anus (Exposed)": "ANUS_EXPOSED",
    "Anus (Covered)": "ANUS_COVERED",
    "Belly (Exposed)": "BELLY_EXPOSED",
    "Belly (Covered)": "BELLY_COVERED",
    "Feet (Exposed)": "FEET_EXPOSED",
    "Feet (Covered)": "FEET_COVERED",
    "Armpits (Exposed)": "ARMPITS_EXPOSED",
    "Armpits (Covered)": "ARMPITS_COVERED",
}

DEFAULT_CHECKED = [
    "Breast (Exposed)",
    "Female Genitalia (Exposed)",
    "Male Genitalia (Exposed)",
    "Buttocks (Exposed)",
    "Anus (Exposed)",
]


def get_gpu_provider():
    try:
        providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def apply_mosaic(image, x1, y1, x2, y2, block_size=10):
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return image
    h, w = region.shape[:2]
    small = cv2.resize(region, (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = mosaic
    return image


class CensorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Censor")
        self.root.geometry("620x700")
        self.root.resizable(False, False)

        self.detector = None
        self.processing = False

        self._build_ui()
        self._init_detector()

    def _init_detector(self):
        providers = get_gpu_provider()
        provider_names = []
        for p in providers:
            if isinstance(p, tuple):
                provider_names.append(p[0])
            else:
                provider_names.append(p)

        try:
            self.detector = NudeDetector(providers=providers)
            if "CUDAExecutionProvider" in provider_names:
                self.gpu_label.config(text="GPU: CUDA (NVIDIA)", foreground="green")
            else:
                self.gpu_label.config(text="GPU: Not available (using CPU)", foreground="red")
        except Exception:
            # CUDA init failed, fall back to CPU
            self.detector = NudeDetector(providers=["CPUExecutionProvider"])
            self.gpu_label.config(text="GPU: Failed to init, using CPU", foreground="orange")

    def _build_ui(self):
        # --- File selection ---
        file_frame = ttk.LabelFrame(self.root, text="Input", padding=10)
        file_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=55).pack(side="left", padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self._browse_file).pack(side="left")

        # --- GPU status ---
        self.gpu_label = ttk.Label(self.root, text="GPU: Detecting...", font=("Segoe UI", 9))
        self.gpu_label.pack(anchor="w", padx=15)

        # --- Body parts selection ---
        parts_frame = ttk.LabelFrame(self.root, text="What to censor", padding=10)
        parts_frame.pack(fill="x", padx=10, pady=5)

        self.class_vars = {}
        row = 0
        col = 0
        for label, class_name in ALL_CLASSES.items():
            var = tk.BooleanVar(value=label in DEFAULT_CHECKED)
            self.class_vars[label] = var
            cb = ttk.Checkbutton(parts_frame, text=label, variable=var)
            cb.grid(row=row, column=col, sticky="w", padx=5, pady=2)
            col += 1
            if col >= 3:
                col = 0
                row += 1

        # Select all / none buttons
        btn_row = ttk.Frame(parts_frame)
        btn_row.grid(row=row + 1, column=0, columnspan=3, pady=(5, 0))
        ttk.Button(btn_row, text="Select All", command=self._select_all).pack(side="left", padx=5)
        ttk.Button(btn_row, text="Select None", command=self._select_none).pack(side="left", padx=5)
        ttk.Button(btn_row, text="Exposed Only", command=self._select_exposed).pack(side="left", padx=5)

        # --- Mosaic settings ---
        mosaic_frame = ttk.LabelFrame(self.root, text="Mosaic Settings", padding=10)
        mosaic_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(mosaic_frame, text="Block Size:").pack(anchor="w")

        slider_frame = ttk.Frame(mosaic_frame)
        slider_frame.pack(fill="x")

        self.block_size = tk.IntVar(value=10)
        self.block_slider = ttk.Scale(slider_frame, from_=3, to=40, variable=self.block_size,
                                      orient="horizontal", command=self._update_block_label)
        self.block_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.block_label = ttk.Label(slider_frame, text="10", width=4)
        self.block_label.pack(side="left")

        ttk.Label(mosaic_frame, text="(Higher = more pixelated)", font=("Segoe UI", 8)).pack(anchor="w")

        # --- Confidence threshold ---
        conf_frame = ttk.LabelFrame(self.root, text="Detection Confidence", padding=10)
        conf_frame.pack(fill="x", padx=10, pady=5)

        conf_slider_frame = ttk.Frame(conf_frame)
        conf_slider_frame.pack(fill="x")

        self.confidence = tk.DoubleVar(value=0.4)
        self.conf_slider = ttk.Scale(conf_slider_frame, from_=0.1, to=0.9, variable=self.confidence,
                                     orient="horizontal", command=self._update_conf_label)
        self.conf_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.conf_label = ttk.Label(conf_slider_frame, text="0.40", width=5)
        self.conf_label.pack(side="left")

        ttk.Label(conf_frame, text="(Lower = more aggressive detection, may have false positives)",
                  font=("Segoe UI", 8)).pack(anchor="w")

        # --- Action buttons ---
        action_frame = ttk.Frame(self.root, padding=10)
        action_frame.pack(fill="x", padx=10)

        self.run_btn = ttk.Button(action_frame, text="Censor!", command=self._run_censor)
        self.run_btn.pack(fill="x", ipady=8)

        # --- Progress ---
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(self.root, textvariable=self.progress_var, font=("Segoe UI", 9))
        self.progress_label.pack(anchor="w", padx=15, pady=(0, 5))

        self.progress_bar = ttk.Progressbar(self.root, mode="determinate")
        self.progress_bar.pack(fill="x", padx=15, pady=(0, 10))

    def _update_block_label(self, val):
        self.block_label.config(text=str(int(float(val))))

    def _update_conf_label(self, val):
        self.conf_label.config(text=f"{float(val):.2f}")

    def _browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("All supported", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mkv *.mov *.wmv"),
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Videos", "*.mp4 *.avi *.mkv *.mov *.wmv"),
            ]
        )
        if path:
            self.file_path.set(path)

    def _select_all(self):
        for var in self.class_vars.values():
            var.set(True)

    def _select_none(self):
        for var in self.class_vars.values():
            var.set(False)

    def _select_exposed(self):
        for label, var in self.class_vars.items():
            var.set("Exposed" in label)

    def _get_selected_classes(self):
        return [ALL_CLASSES[label] for label, var in self.class_vars.items() if var.get()]

    def _is_video(self, path):
        ext = os.path.splitext(path)[1].lower()
        return ext in (".mp4", ".avi", ".mkv", ".mov", ".wmv")

    def _run_censor(self):
        if self.processing:
            return

        input_path = self.file_path.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid file.")
            return

        classes = self._get_selected_classes()
        if not classes:
            messagebox.showerror("Error", "Please select at least one body part to censor.")
            return

        self.processing = True
        self.run_btn.config(state="disabled")

        thread = threading.Thread(target=self._process, args=(input_path, classes), daemon=True)
        thread.start()

    def _process(self, input_path, classes):
        try:
            block_size = int(self.block_size.get())
            min_confidence = float(self.confidence.get())
            name, ext = os.path.splitext(input_path)
            output_path = f"{name}_censored{ext}"

            if self._is_video(input_path):
                self._process_video(input_path, output_path, classes, block_size, min_confidence)
            else:
                self._update_status("Processing image...")
                self._censor_frame(input_path, output_path, classes, block_size, min_confidence)
                self._update_progress(100)
                self._update_status(f"Done! Saved: {output_path}")

            messagebox.showinfo("Done", f"Saved to:\n{output_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._update_status(f"Error: {e}")

        finally:
            self.processing = False
            self.root.after(0, lambda: self.run_btn.config(state="normal"))

    def _censor_frame(self, image_path, output_path, classes, block_size, min_confidence):
        detections = self.detector.detect(image_path)
        image = cv2.imread(image_path)

        for det in detections:
            if det["class"] in classes and det["score"] >= min_confidence:
                x, y, w, h = [int(c) for c in det["box"]]
                image = apply_mosaic(image, x, y, x + w, y + h, block_size)

        cv2.imwrite(output_path, image)

    def _process_video(self, input_path, output_path, classes, block_size, min_confidence):
        frames_dir = tempfile.mkdtemp(prefix="frames_")
        censored_dir = tempfile.mkdtemp(prefix="censored_")

        try:
            self._update_status("Extracting frames...")
            subprocess.run([
                FFMPEG, "-i", input_path,
                "-qscale:v", "2",
                os.path.join(frames_dir, "frame_%06d.jpg")
            ], check=True, capture_output=True)

            frames = sorted(Path(frames_dir).glob("*.jpg"))
            total = len(frames)

            for i, frame in enumerate(frames, 1):
                out_frame = os.path.join(censored_dir, frame.name)
                self._censor_frame(str(frame), out_frame, classes, block_size, min_confidence)
                pct = int(i / total * 90)
                self._update_progress(pct)
                self._update_status(f"Processing frame {i}/{total}...")

            fps_result = subprocess.run([
                FFPROBE, "-v", "0", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0", input_path
            ], capture_output=True, text=True)
            fps = fps_result.stdout.strip()

            self._update_status("Reassembling video...")
            subprocess.run([
                FFMPEG, "-y",
                "-framerate", fps,
                "-i", os.path.join(censored_dir, "frame_%06d.jpg"),
                "-i", input_path,
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "copy",
                "-pix_fmt", "yuv420p",
                output_path
            ], check=True, capture_output=True)

            self._update_progress(100)
            self._update_status(f"Done! Saved: {output_path}")

        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)
            shutil.rmtree(censored_dir, ignore_errors=True)

    def _update_status(self, text):
        self.root.after(0, lambda: self.progress_var.set(text))

    def _update_progress(self, value):
        self.root.after(0, lambda: self.progress_bar.configure(value=value))


if __name__ == "__main__":
    root = tk.Tk()
    app = CensorApp(root)
    root.mainloop()
