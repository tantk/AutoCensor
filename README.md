# AutoCensor

Auto-censor exposed body parts in images and videos using AI.

Uses [NudeNet](https://github.com/notAI-tech/NudeNet) for detection and applies pixelated mosaic censoring.

## Features

- **Auto-detection** of 18 body part classes (breasts, genitalia, face, etc.)
- **Selectable body parts** via checkboxes
- **Adjustable mosaic intensity** with slider
- **Confidence threshold** to control detection sensitivity
- **Video timeline sliders** - set start/end time for censoring
- **Trim mode** - output only the selected video range
- **Image & video support** (jpg, png, bmp, webp, mp4, avi, mkv, mov, wmv)
- Bundled ffmpeg - no external dependencies

## Download

Download the latest release from the [Releases](https://github.com/tantk/AutoCensor/releases) page.

Unzip and run `AutoCensor.exe`. No installation needed.

## Usage

1. Click **Browse** and select an image or video
2. Check the body parts you want to censor
3. Adjust mosaic block size and confidence as needed
4. For videos, use the timeline sliders to set the range
5. Click **Censor!**

Output is saved next to the input file as `filename_censored.ext`.

## Run from source

```bash
pip install nudenet onnxruntime opencv-python-headless
python censor_app.py
```

ffmpeg must be on PATH for video processing.

## Build exe

```bash
pip install pyinstaller
pyinstaller censor_app.spec
```

Copy `ffmpeg.exe` into the `dist/AutoCensor/` folder.
