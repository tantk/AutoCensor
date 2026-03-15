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
import onnxruntime
from nudenet import NudeDetector

# Use bundled ffmpeg if available
APP_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
_ffmpeg_local = os.path.join(APP_DIR, "ffmpeg.exe")
FFMPEG = _ffmpeg_local if os.path.exists(_ffmpeg_local) else "ffmpeg"


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
        self.root.geometry("620x780")
        self.root.resizable(False, False)

        self.detector = None
        self.processing = False

        self._build_ui()
        self._init_detector()

    def _init_detector(self):
        self.detector = NudeDetector(providers=["CPUExecutionProvider"])

    def _build_ui(self):
        # --- File selection ---
        file_frame = ttk.LabelFrame(self.root, text="Input", padding=10)
        file_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=55).pack(side="left", padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self._browse_file).pack(side="left")


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

        # --- Video time range ---
        self.time_frame = ttk.LabelFrame(self.root, text="Video Time Range", padding=10)
        self.time_frame.pack(fill="x", padx=10, pady=5)

        # Start slider
        start_row = ttk.Frame(self.time_frame)
        start_row.pack(fill="x", pady=(0, 3))
        ttk.Label(start_row, text="Start:", width=5).pack(side="left")
        self.start_val = tk.DoubleVar(value=0)
        self.start_slider = ttk.Scale(start_row, from_=0, to=100, variable=self.start_val,
                                      orient="horizontal", command=self._update_start_label)
        self.start_slider.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.start_label = ttk.Label(start_row, text="00:00", width=8)
        self.start_label.pack(side="left")

        # End slider
        end_row = ttk.Frame(self.time_frame)
        end_row.pack(fill="x")
        ttk.Label(end_row, text="End:", width=5).pack(side="left")
        self.end_val = tk.DoubleVar(value=100)
        self.end_slider = ttk.Scale(end_row, from_=0, to=100, variable=self.end_val,
                                    orient="horizontal", command=self._update_end_label)
        self.end_slider.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.end_label = ttk.Label(end_row, text="00:00", width=8)
        self.end_label.pack(side="left")

        self.video_duration = 0
        self.video_fps = 30

        # Trim checkbox
        self.trim_video = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.time_frame, text="Output selected range only (trim)",
                        variable=self.trim_video).pack(anchor="w", pady=(5, 0))

        ttk.Label(self.time_frame, text="(Load a video to enable. Drag sliders to set range.)",
                  font=("Segoe UI", 8)).pack(anchor="w", pady=(3, 0))
        # Disable until video loaded
        self.start_slider.config(state="disabled")
        self.end_slider.config(state="disabled")

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

    def _format_time(self, seconds):
        seconds = max(0, seconds)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _update_start_label(self, val):
        sec = float(val) / 100 * self.video_duration
        self.start_label.config(text=self._format_time(sec))

    def _update_end_label(self, val):
        sec = float(val) / 100 * self.video_duration
        self.end_label.config(text=self._format_time(sec))

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
            if self._is_video(path):
                cap = cv2.VideoCapture(path)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.video_duration = frame_count / self.video_fps
                cap.release()

                self.start_slider.config(state="normal")
                self.end_slider.config(state="normal")
                self.start_val.set(0)
                self.end_val.set(100)
                self.start_label.config(text=self._format_time(0))
                self.end_label.config(text=self._format_time(self.video_duration))
                self.time_frame.config(text=f"Video Time Range ({self._format_time(self.video_duration)} total)")
            else:
                self.start_slider.config(state="disabled")
                self.end_slider.config(state="disabled")
                self.video_duration = 0
                self.time_frame.config(text="Video Time Range")

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
                start_pct = self.start_val.get()
                end_pct = self.end_val.get()
                start_sec = start_pct / 100 * self.video_duration if start_pct > 0 else None
                end_sec = end_pct / 100 * self.video_duration if end_pct < 100 else None
                trim = self.trim_video.get()
                self._process_video(input_path, output_path, classes, block_size, min_confidence, start_sec, end_sec, trim)
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

    def _process_video(self, input_path, output_path, classes, block_size, min_confidence, start_sec=None, end_sec=None, trim=False):
        frames_dir = tempfile.mkdtemp(prefix="frames_")
        censored_dir = tempfile.mkdtemp(prefix="censored_")

        try:
            # Extract frames (trimmed or full)
            self._update_status("Extracting frames...")
            extract_cmd = [FFMPEG]
            if trim and start_sec is not None:
                extract_cmd += ["-ss", str(start_sec)]
            if trim and end_sec is not None:
                duration = end_sec - (start_sec or 0)
                extract_cmd += ["-t", str(duration)]
            extract_cmd += ["-i", input_path, "-qscale:v", "2",
                           os.path.join(frames_dir, "frame_%06d.jpg")]
            subprocess.run(extract_cmd, check=True, capture_output=True)

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            frames = sorted(Path(frames_dir).glob("*.jpg"))
            total = len(frames)

            if trim:
                # All extracted frames are in range
                for i, frame in enumerate(frames, 1):
                    out_frame = os.path.join(censored_dir, frame.name)
                    if classes:
                        self._censor_frame(str(frame), out_frame, classes, block_size, min_confidence)
                        self._update_status(f"Censoring frame {i}/{total}...")
                    else:
                        shutil.copy2(str(frame), out_frame)
                        self._update_status(f"Processing frame {i}/{total}...")
                    self._update_progress(int(i / total * 90))
            else:
                # Full video — censor only the range
                start_frame = int(start_sec * fps) + 1 if start_sec is not None else 1
                end_frame = int(end_sec * fps) + 1 if end_sec is not None else float('inf')

                for i, frame in enumerate(frames, 1):
                    out_frame = os.path.join(censored_dir, frame.name)
                    if start_frame <= i <= end_frame and classes:
                        self._censor_frame(str(frame), out_frame, classes, block_size, min_confidence)
                        self._update_status(f"Censoring frame {i}/{total}...")
                    else:
                        shutil.copy2(str(frame), out_frame)
                        self._update_status(f"Copying frame {i}/{total}...")
                    self._update_progress(int(i / total * 90))

            self._update_status("Reassembling video...")
            reassemble_cmd = [
                FFMPEG, "-y",
                "-framerate", str(fps),
                "-i", os.path.join(censored_dir, "frame_%06d.jpg"),
            ]

            if trim and (start_sec is not None or end_sec is not None):
                # Extract matching audio segment
                audio_file = os.path.join(frames_dir, "audio.aac")
                audio_cmd = [FFMPEG, "-y", "-i", input_path]
                if start_sec is not None:
                    audio_cmd += ["-ss", str(start_sec)]
                if end_sec is not None:
                    audio_cmd += ["-t", str(end_sec - (start_sec or 0))]
                audio_cmd += ["-vn", "-acodec", "copy", audio_file]
                subprocess.run(audio_cmd, capture_output=True)

                if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                    reassemble_cmd += ["-i", audio_file, "-map", "0:v", "-map", "1:a"]
                else:
                    reassemble_cmd += ["-map", "0:v"]
            else:
                reassemble_cmd += ["-i", input_path, "-map", "0:v", "-map", "1:a?"]

            reassemble_cmd += [
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "copy",
                "-pix_fmt", "yuv420p",
                "-shortest",
                output_path
            ]
            subprocess.run(reassemble_cmd, check=True, capture_output=True)

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
