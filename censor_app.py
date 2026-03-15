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
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe import Image, ImageFormat

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




def apply_mosaic(image, x1, y1, x2, y2, block_size=10, expand=0.3):
    """Apply mosaic with optional region expansion (0.3 = 30% bigger each side)."""
    img_h, img_w = image.shape[:2]
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * expand)
    pad_y = int(h * expand)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return image
    rh, rw = region.shape[:2]
    small = cv2.resize(region, (max(1, rw // block_size), max(1, rh // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = mosaic
    return image


class CensorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Censor")
        self.root.geometry("620x950")
        self.root.resizable(False, False)

        self.detector = None
        self.processing = False

        self._build_ui()
        self._init_detector()

    def _get_model_path(self, model_name):
        """Find model file bundled or next to exe/script."""
        if model_name == "320n (fast)":
            return None  # Use default bundled model
        # Look for 640m.onnx next to exe or in models/ folder
        for search_dir in [APP_DIR, os.path.join(APP_DIR, "models"),
                           os.path.join(APP_DIR, "_internal"), os.path.join(APP_DIR, "_internal", "models")]:
            path = os.path.join(search_dir, "640m.onnx")
            if os.path.exists(path):
                return path
        return None

    def _find_pose_model(self):
        for search_dir in [APP_DIR, os.path.join(APP_DIR, "models"),
                           os.path.join(APP_DIR, "_internal"), os.path.join(APP_DIR, "_internal", "models")]:
            path = os.path.join(search_dir, "pose_landmarker_lite.task")
            if os.path.exists(path):
                return path
        return None

    def _init_detector(self):
        model = self.model_var.get() if hasattr(self, 'model_var') else "320n (fast)"
        model_path = self._get_model_path(model)
        if model_path:
            self.detector = NudeDetector(model_path=model_path, inference_resolution=640,
                                         providers=["CPUExecutionProvider"])
        else:
            self.detector = NudeDetector(providers=["CPUExecutionProvider"])

        # Init pose landmarker
        pose_model = self._find_pose_model()
        if pose_model:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model),
                running_mode=RunningMode.IMAGE,
                num_poses=5
            )
            self.pose_landmarker = PoseLandmarker.create_from_options(options)
        else:
            self.pose_landmarker = None

    def _build_ui(self):
        # --- File selection ---
        file_frame = ttk.LabelFrame(self.root, text="Input", padding=10)
        file_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=55).pack(side="left", padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self._browse_file).pack(side="left")

        # --- Model selection ---
        model_frame = ttk.LabelFrame(self.root, text="Detection Model", padding=10)
        model_frame.pack(fill="x", padx=10, pady=(5, 5))

        model_row = ttk.Frame(model_frame)
        model_row.pack(fill="x")

        self.model_var = tk.StringVar(value="320n (fast)")
        models = ["320n (fast)", "640m (accurate)"]
        ttk.Label(model_row, text="Model:").pack(side="left")
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, values=models,
                                        state="readonly", width=20)
        self.model_combo.pack(side="left", padx=(5, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        self.model_status = ttk.Label(model_row, text="", font=("Segoe UI", 8))
        self.model_status.pack(side="left")

        ttk.Label(model_frame, text="640m is slower but more accurate. Place 640m.onnx next to AutoCensor.exe.",
                  font=("Segoe UI", 8)).pack(anchor="w")

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

        # --- Pose-based censoring ---
        pose_frame = ttk.LabelFrame(self.root, text="Pose-Based Censoring (body tracking)", padding=10)
        pose_frame.pack(fill="x", padx=10, pady=5)

        self.pose_crotch = tk.BooleanVar(value=False)
        ttk.Checkbutton(pose_frame, text="Censor crotch area (tracks hip skeleton)",
                        variable=self.pose_crotch).pack(anchor="w")

        self.pose_chest = tk.BooleanVar(value=False)
        ttk.Checkbutton(pose_frame, text="Censor chest area (tracks shoulder skeleton)",
                        variable=self.pose_chest).pack(anchor="w")

        self.pose_butt = tk.BooleanVar(value=False)
        ttk.Checkbutton(pose_frame, text="Censor butt area (tracks hip skeleton, rear)",
                        variable=self.pose_butt).pack(anchor="w")

        ttk.Label(pose_frame, text="Uses body skeleton tracking — more reliable than AI detection for constant exposure.",
                  font=("Segoe UI", 8)).pack(anchor="w", pady=(3, 0))

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

    def _on_model_change(self, event=None):
        model = self.model_var.get()
        if model == "640m (accurate)":
            path = self._get_model_path(model)
            if not path:
                self.model_status.config(text="640m.onnx not found!", foreground="red")
                self.model_var.set("320n (fast)")
                return
        self.model_status.config(text="Loading...", foreground="gray")
        self.root.update()
        self._init_detector()
        self.model_status.config(text="Ready", foreground="green")

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

    def _get_all_poses(self, image_path):
        """Get all pose skeletons from image. Returns list of (hip_center, pose_data) tuples."""
        if not self.pose_landmarker:
            return []
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = self.pose_landmarker.detect(mp_image)

        poses = []
        for pose in results.pose_landmarks:
            left_hip = pose[23]
            right_hip = pose[24]
            cx = (left_hip.x + right_hip.x) / 2 * w
            cy = (left_hip.y + right_hip.y) / 2 * h
            poses.append({"cx": cx, "cy": cy, "landmarks": pose, "w": w, "h": h})
        return poses

    def _pose_to_boxes(self, pose_data):
        """Convert a pose skeleton to censor boxes based on selected regions."""
        boxes = []
        pose = pose_data["landmarks"]
        w, h = pose_data["w"], pose_data["h"]

        left_hip = pose[23]
        right_hip = pose[24]
        left_knee = pose[25]
        left_shoulder = pose[11]
        right_shoulder = pose[12]

        do_crotch = self.pose_crotch.get()
        do_chest = self.pose_chest.get()
        do_butt = self.pose_butt.get()

        if do_crotch or do_butt:
            cx = int((left_hip.x + right_hip.x) / 2 * w)
            cy = int((left_hip.y + right_hip.y) / 2 * h)
            hip_width = max(abs(int((left_hip.x - right_hip.x) * w)), 30)
            thigh_len = max(abs(int((left_knee.y - left_hip.y) * h)), 40)
            pad = int(hip_width * 0.4)

            if do_crotch:
                x1 = cx - hip_width // 2 - pad
                y1 = cy - pad
                x2 = cx + hip_width // 2 + pad
                y2 = cy + int(thigh_len * 0.5) + pad
                boxes.append([max(0, x1), max(0, y1), min(w, x2) - max(0, x1), min(h, y2) - max(0, y1)])

            if do_butt:
                x1 = cx - hip_width // 2 - pad
                y1 = cy - int(thigh_len * 0.2)
                x2 = cx + hip_width // 2 + pad
                y2 = cy + int(thigh_len * 0.4) + pad
                boxes.append([max(0, x1), max(0, y1), min(w, x2) - max(0, x1), min(h, y2) - max(0, y1)])

        if do_chest:
            scx = int((left_shoulder.x + right_shoulder.x) / 2 * w)
            scy = int((left_shoulder.y + right_shoulder.y) / 2 * h)
            shoulder_width = max(abs(int((left_shoulder.x - right_shoulder.x) * w)), 30)
            torso_len = max(abs(int((left_hip.y - left_shoulder.y) * h)), 40)
            spad = int(shoulder_width * 0.2)

            x1 = scx - shoulder_width // 2 - spad
            y1 = scy - spad
            x2 = scx + shoulder_width // 2 + spad
            y2 = scy + int(torso_len * 0.5)
            boxes.append([max(0, x1), max(0, y1), min(w, x2) - max(0, x1), min(h, y2) - max(0, y1)])

        return boxes

    def _get_pose_boxes(self, image_path):
        """Simple mode: censor all detected poses (used for images)."""
        poses = self._get_all_poses(image_path)
        boxes = []
        for p in poses:
            boxes += self._pose_to_boxes(p)
        return boxes

    def _match_nudenet_to_pose(self, nudenet_boxes, poses, img_w, img_h):
        """Find which poses have a NudeNet detection overlapping their hip area.
        Returns set of pose indices that are flagged as nude."""
        flagged = set()
        for nb in nudenet_boxes:
            nx, ny, nw, nh = nb
            ncx = nx + nw / 2
            ncy = ny + nh / 2
            best_dist = float('inf')
            best_idx = -1
            for idx, p in enumerate(poses):
                dist = ((p["cx"] - ncx) ** 2 + (p["cy"] - ncy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            # Match if detection is within reasonable distance of hip center
            threshold = max(img_w, img_h) * 0.2
            if best_idx >= 0 and best_dist < threshold:
                flagged.add(best_idx)
        return flagged

    def _find_closest_pose(self, prev_cx, prev_cy, poses, scene_threshold):
        """Find the pose closest to previous position. Returns index or -1 if scene change."""
        best_dist = float('inf')
        best_idx = -1
        for idx, p in enumerate(poses):
            dist = ((p["cx"] - prev_cx) ** 2 + (p["cy"] - prev_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0 and best_dist < scene_threshold:
            return best_idx
        return -1  # Scene change or lost tracking

    def _detect_frame(self, image_path, classes, min_confidence):
        """Return list of [x, y, w, h] boxes for matching detections."""
        detections = self.detector.detect(image_path)
        boxes = []
        for det in detections:
            if det["class"] in classes and det["score"] >= min_confidence:
                boxes.append([int(c) for c in det["box"]])
        return boxes

    def _censor_frame(self, image_path, output_path, classes, block_size, min_confidence):
        boxes = self._detect_frame(image_path, classes, min_confidence)
        pose_boxes = self._get_pose_boxes(image_path)
        self._apply_boxes(image_path, output_path, boxes + pose_boxes, block_size)

    def _apply_boxes(self, image_path, output_path, boxes, block_size):
        image = cv2.imread(image_path)
        for x, y, w, h in boxes:
            image = apply_mosaic(image, x, y, x + w, y + h, block_size)
        cv2.imwrite(output_path, image)

    def _smooth_detections(self, all_boxes, window=5):
        """Fill gaps in detections. If frame N has no boxes but nearby frames do, interpolate."""
        smoothed = [list(b) for b in all_boxes]
        total = len(smoothed)

        for i in range(total):
            if smoothed[i]:
                continue
            # Look backward and forward for nearest detections
            prev_boxes = None
            next_boxes = None
            for back in range(1, window + 1):
                if i - back >= 0 and all_boxes[i - back]:
                    prev_boxes = all_boxes[i - back]
                    break
            for fwd in range(1, window + 1):
                if i + fwd < total and all_boxes[i + fwd]:
                    next_boxes = all_boxes[i + fwd]
                    break
            # If both neighbors have detections, use the closer one
            if prev_boxes and next_boxes:
                smoothed[i] = prev_boxes
            elif prev_boxes:
                smoothed[i] = prev_boxes
            elif next_boxes:
                smoothed[i] = next_boxes

        return smoothed

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

            # Determine which frames to censor
            if trim:
                censor_start = 1
                censor_end = total
            else:
                censor_start = int(start_sec * fps) + 1 if start_sec is not None else 1
                censor_end = int(end_sec * fps) + 1 if end_sec is not None else total

            use_pose = self.pose_crotch.get() or self.pose_chest.get() or self.pose_butt.get()

            # Smart tracking: NudeNet triggers, pose takes over
            # tracked_people: dict of track_id -> {"cx": float, "cy": float}
            tracked_people = {}
            next_track_id = 0
            scene_threshold = max(1920, 1080) * 0.15  # 15% of frame = scene change

            for i, frame in enumerate(frames, 1):
                out_frame = os.path.join(censored_dir, frame.name)
                in_range = censor_start <= i <= censor_end

                if not in_range:
                    shutil.copy2(str(frame), out_frame)
                    self._update_progress(int(i / total * 90))
                    continue

                frame_path = str(frame)
                all_frame_boxes = []

                # Step 1: NudeNet detection
                if classes:
                    nudenet_boxes = self._detect_frame(frame_path, classes, min_confidence)
                    all_frame_boxes += nudenet_boxes
                else:
                    nudenet_boxes = []

                # Step 2: Pose tracking
                if use_pose:
                    poses = self._get_all_poses(frame_path)
                    img = cv2.imread(frame_path)
                    img_h, img_w = img.shape[:2] if img is not None else (1080, 1920)

                    # Match NudeNet detections to poses — flag new nude people
                    if nudenet_boxes and poses:
                        flagged_indices = self._match_nudenet_to_pose(nudenet_boxes, poses, img_w, img_h)
                        for idx in flagged_indices:
                            p = poses[idx]
                            # Check if already tracked
                            already_tracked = False
                            for tid, tp in tracked_people.items():
                                dist = ((tp["cx"] - p["cx"]) ** 2 + (tp["cy"] - p["cy"]) ** 2) ** 0.5
                                if dist < scene_threshold:
                                    already_tracked = True
                                    break
                            if not already_tracked:
                                tracked_people[next_track_id] = {"cx": p["cx"], "cy": p["cy"]}
                                next_track_id += 1

                    # Update tracked people positions and generate pose boxes
                    new_tracked = {}
                    for tid, tp in tracked_people.items():
                        match_idx = self._find_closest_pose(tp["cx"], tp["cy"], poses, scene_threshold)
                        if match_idx >= 0:
                            p = poses[match_idx]
                            new_tracked[tid] = {"cx": p["cx"], "cy": p["cy"]}
                            all_frame_boxes += self._pose_to_boxes(p)
                    tracked_people = new_tracked

                # Apply all boxes
                if all_frame_boxes:
                    self._apply_boxes(frame_path, out_frame, all_frame_boxes, block_size)
                    self._update_status(f"Censoring frame {i}/{total} ({len(tracked_people)} tracked)...")
                else:
                    shutil.copy2(frame_path, out_frame)
                    self._update_status(f"Frame {i}/{total}...")

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
