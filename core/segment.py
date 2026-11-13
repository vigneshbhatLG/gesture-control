"""Step 2: find the hand inside the ROI. Classical CV only, no ML.

Two cues:
  - difference: anything that differs from the learned empty background
  - skin colour: pixels inside a YCrCb skin range
mode = "diff" | "skin" | "both" (both = must differ AND be skin-coloured)

Every call here exists 1:1 in OpenCV C++, so this file ports directly.
"""
import cv2
import numpy as np

MODES = ("diff", "skin", "both")


class HandSegmenter:
    def __init__(self, cfg):
        s = cfg["segment"]
        self.mode = s["mode"]
        self.thresh = s["diffThreshold"]
        self.min_area_ratio = s["minAreaRatio"]
        self.bg_frames = s["bgFrames"]
        self.max_fg_ratio = s.get("maxFgRatio", 0.7)
        self.status = "no hand"
        self.debug = None
        sk = s["skin"]
        self.skin_lo = np.array([0, sk["crMin"], sk["cbMin"]], np.uint8)
        self.skin_hi = np.array([255, sk["crMax"], sk["cbMax"]], np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.reset_background()

    def reset_background(self):
        self.bg = None
        self.seen = 0

    @property
    def calibrating(self):
        return self.seen < self.bg_frames

    def next_mode(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]

    def process(self, roi, exclude=()):
        """roi: BGR image of the gesture zone.
        exclude: rects (x, y, w, h) in ROI coords to blank out (e.g. the face).
        Returns (mask, hand_contour or None)."""
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (7, 7), 0)

        # 1. Learn the empty background for the first N frames
        if self.bg is None:
            self.bg = gray.astype(np.float32)
        if self.calibrating:
            cv2.accumulateWeighted(gray, self.bg, 0.5)
            self.seen += 1
            return np.zeros_like(gray), None

        # 2. Difference from background, with brightness compensation.
        #    Webcam auto-exposure brightens/darkens the WHOLE image when
        #    something moves (your head, your hand). The median of the signed
        #    difference = that global shift (as long as the hand covers < half
        #    the zone), so we subtract it out.
        signed = gray.astype(np.float32) - self.bg
        shift = float(np.median(signed))
        diff = cv2.convertScaleAbs(signed - shift)
        _, motion = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)

        # 3. Skin colour (YCrCb is less sensitive to brightness than RGB)
        skin = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb),
                           self.skin_lo, self.skin_hi)

        self.debug = (motion, skin)  # raw cues, for the debug view

        if self.mode == "diff":
            mask = motion
        elif self.mode == "skin":
            mask = skin
        else:
            mask = cv2.bitwise_and(motion, skin)

        # Blank out excluded regions (face). Clip manually: negative numpy
        # indices would wrap around to the other side of the image.
        H, W = mask.shape
        for ex, ey, ew, eh in exclude:
            x0, y0 = max(ex, 0), max(ey, 0)
            x1, y1 = min(ex + ew, W), min(ey + eh, H)
            if x0 < x1 and y0 < y1:
                mask[y0:y1, x0:x1] = 0

        # 4. Clean up: remove specks, fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # 5. Sanity check: if most of the zone "changed", that's a lighting
        #    change, not a hand. Re-learn quickly instead of reporting a hand
        #    (otherwise we'd get stuck forever: hand present -> never adapt).
        fg_ratio = cv2.countNonZero(mask) / mask.size
        if fg_ratio > self.max_fg_ratio:
            self.status = "lighting change"
            cv2.accumulateWeighted(gray, self.bg, 0.3)
            return mask, None

        # 6. Biggest blob = hand (if it is big enough)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        hand = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) >= self.min_area_ratio * mask.size:
                hand = c

        # 7. No hand -> slowly adapt background to lighting drift.
        #    Never adapt while a hand is present, or the hand fades into it.
        if hand is None:
            cv2.accumulateWeighted(gray, self.bg, 0.05)

        self.status = "hand" if hand is not None else "no hand"
        return mask, hand
