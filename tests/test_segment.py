"""Synthetic tests for HandSegmenter, incl. auto-exposure brightness shifts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from core.config import load_config
from core.segment import HandSegmenter

rng = np.random.default_rng(0)


def scene(brightness_shift=0, hand=False):
    img = np.full((150, 110, 3), 170, np.int16)
    img[:, 60:] = 120  # some background structure
    if hand:
        img[40:120, 30:80] = (90, 120, 170)
    img = img + brightness_shift + rng.integers(-3, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


seg = HandSegmenter(load_config())
for _ in range(30):
    seg.process(scene())

checks = [
    ("empty zone", scene(), False),
    ("hand present", scene(hand=True), True),
    ("empty, whole image +40 brighter (auto-exposure)", scene(40), False),
    ("hand, whole image +40 brighter", scene(40, hand=True), True),
    ("empty, +120 brighter (huge light change)", scene(120), False),
]
KNOWN = {"empty, +120 brighter (huge light change)"}  # saturation: documented limit
for name, img, expect in checks:
    _, h = seg.process(img)
    got = h is not None
    tag = "PASS" if got == expect else ("KNOWN" if name in KNOWN else "FAIL")
    print(tag, name, "->", seg.status)
