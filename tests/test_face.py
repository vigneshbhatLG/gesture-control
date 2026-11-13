"""Check face cascade loads, padding clips to frame, and exclusion blanks the mask.
(Classical version: core/face.py + core/segment.py.)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from core.config import load_config
from core.face import FaceMasker, rects_to_roi
from core.segment import HandSegmenter

fails = 0


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print("PASS" if cond else "FAIL", name, extra)


cfg = load_config()
fm = FaceMasker(cfg)
check("Haar face cascade loads", not fm.cascade.empty())
check("no face in a blank frame", fm.update(np.zeros((240, 320, 3), np.uint8)) == [])
check("padding clipped to frame", fm._pad((280, 5, 40, 40), (240, 320, 3)) == (270, 0, 50, 61))

seg = HandSegmenter(cfg)
bg = np.full((150, 110, 3), 170, np.uint8)
for _ in range(30):
    seg.process(bg)
frame = bg.copy()
frame[40:120, 30:80] = (90, 120, 170)
_, h = seg.process(frame)
check("blob without exclusion -> hand", h is not None)
# exclusion rect partially outside the ROI (negative x) must still work
_, h = seg.process(frame, exclude=[(-20, 20, 120, 120)])
check("blob inside face rect -> no hand", h is None)
check("frame rect -> ROI coords", rects_to_roi([(200, 30, 50, 60)], 201, 19) == [(-1, 11, 50, 60)])

print("ALL PASS" if not fails else f"{fails} FAILED")
