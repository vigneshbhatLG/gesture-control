"""Download pretrained model files into models/ (not committed to git).

- hand_landmarker.task  MediaPipe hands (palm detector + landmarks), used by track.py
- haarcascade_frontalface_default.xml  classical face detector, used by main.py
  (OpenCV 5 contrib doesn't ship this data file)
"""
import urllib.request
from pathlib import Path

MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    "haarcascade_frontalface_default.xml":
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
        "haarcascade_frontalface_default.xml",
}

folder = Path(__file__).resolve().parent.parent / "models"
folder.mkdir(exist_ok=True)
for name, url in MODELS.items():
    dest = folder / name
    if dest.exists():
        print("already present:", dest)
        continue
    urllib.request.urlretrieve(url, dest)
    print(f"downloaded {dest} ({dest.stat().st_size / 1e6:.2f} MB)")
