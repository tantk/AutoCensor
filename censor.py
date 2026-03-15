"""
Auto-censor exposed body parts in images or videos.
Usage:
    censor image input.jpg
    censor video input.mp4
    censor image input.jpg --face --belly --block-size 20
    censor video input.mp4 -o censored.mp4 --face
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import cv2
from nudenet import NudeDetector

detector = NudeDetector()

BODY_PARTS = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
]

FACE_PARTS = ["FACE_FEMALE", "FACE_MALE"]
BELLY_PARTS = ["BELLY_EXPOSED"]
FEET_PARTS = ["FEET_EXPOSED"]
ARMPITS_PARTS = ["ARMPITS_EXPOSED"]


def get_censor_parts(args):
    parts = list(BODY_PARTS)
    if args.face:
        parts += FACE_PARTS
    if args.belly:
        parts += BELLY_PARTS
    if args.feet:
        parts += FEET_PARTS
    if args.armpits:
        parts += ARMPITS_PARTS
    if args.all:
        parts += FACE_PARTS + BELLY_PARTS + FEET_PARTS + ARMPITS_PARTS
    return parts


def apply_mosaic(image, x1, y1, x2, y2, block_size=10):
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return image
    h, w = region.shape[:2]
    small = cv2.resize(region, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = mosaic
    return image


def censor_frame(image_path, output_path, censor_parts, block_size=10):
    detections = detector.detect(image_path)
    image = cv2.imread(image_path)

    for det in detections:
        if det["class"] in censor_parts:
            x, y, w, h = [int(c) for c in det["box"]]
            image = apply_mosaic(image, x, y, x + w, y + h, block_size)

    cv2.imwrite(output_path, image)
    return output_path


def censor_image(input_path, output_path, censor_parts, block_size):
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_censored{ext}"

    censor_frame(input_path, output_path, censor_parts, block_size)
    print(f"Saved: {output_path}")


def censor_video(input_path, output_path, censor_parts, block_size):
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_censored{ext}"

    frames_dir = tempfile.mkdtemp(prefix="frames_")
    censored_dir = tempfile.mkdtemp(prefix="censored_")

    try:
        print("Extracting frames...")
        subprocess.run([
            "ffmpeg", "-i", input_path,
            "-qscale:v", "2",
            os.path.join(frames_dir, "frame_%06d.jpg")
        ], check=True, capture_output=True)

        frames = sorted(Path(frames_dir).glob("*.jpg"))
        total = len(frames)
        print(f"Processing {total} frames...")

        for i, frame in enumerate(frames, 1):
            out_frame = os.path.join(censored_dir, frame.name)
            censor_frame(str(frame), out_frame, censor_parts, block_size)
            if i % 10 == 0 or i == total:
                print(f"  {i}/{total} frames done")

        fps_result = subprocess.run([
            "ffprobe", "-v", "0", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0", input_path
        ], capture_output=True, text=True)
        fps = fps_result.stdout.strip()

        print("Reassembling video...")
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", fps,
            "-i", os.path.join(censored_dir, "frame_%06d.jpg"),
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path
        ], check=True, capture_output=True)

        print(f"Saved: {output_path}")

    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(censored_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-censor exposed body parts in images and videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  censor image photo.jpg                    Body parts only
  censor image photo.jpg --face             Body + face
  censor image photo.jpg --all              Everything
  censor video clip.mp4 --face -b 20        Body + face, heavy mosaic
  censor video clip.mp4 -o out.mp4 --all    Everything, custom output
        """)
    parser.add_argument("mode", choices=["image", "video"])
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--block-size", "-b", type=int, default=10,
                        help="Mosaic block size (default: 10, bigger = more pixelated)")
    parser.add_argument("--face", action="store_true", help="Also censor faces")
    parser.add_argument("--belly", action="store_true", help="Also censor belly")
    parser.add_argument("--feet", action="store_true", help="Also censor feet")
    parser.add_argument("--armpits", action="store_true", help="Also censor armpits")
    parser.add_argument("--all", action="store_true", help="Censor everything (body + face + belly + feet + armpits)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    censor_parts = get_censor_parts(args)
    print(f"Censoring: {', '.join(censor_parts)}")
    print(f"Block size: {args.block_size}")

    if args.mode == "image":
        censor_image(args.input, args.output, censor_parts, args.block_size)
    else:
        censor_video(args.input, args.output, censor_parts, args.block_size)

    input("Press Enter to exit...")
