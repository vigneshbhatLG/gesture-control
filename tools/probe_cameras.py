"""List camera indices that open, with resolution, FPS and a snapshot of each.

Usage: python tools/probe_cameras.py [output_dir]
"""
import sys
import time
from pathlib import Path

import cv2

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)

for idx in range(5):
    for backend, name in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            continue
        frames, t0, frame = 0, time.time(), None
        while time.time() - t0 < 2:
            ok, f = cap.read()
            if ok:
                frame, frames = f, frames + 1
        fps = frames / (time.time() - t0)
        if frame is not None:
            p = out / f"cam{idx}_{name}.jpg"
            cv2.imwrite(str(p), frame)
            print(f"index {idx} {name}: {frame.shape[1]}x{frame.shape[0]} ~{fps:.0f} fps -> {p}")
        else:
            print(f"index {idx} {name}: opened but no frames")
        cap.release()
