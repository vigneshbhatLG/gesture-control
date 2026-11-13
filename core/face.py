"""Face exclusion: classical (non-deep-learning) face detection so the head
is never mistaken for a hand.

Uses OpenCV's built-in Haar cascade (Viola-Jones, 2001): a small pretrained
XML file that ships with OpenCV. No training on our side, fast on CPU, and
cv::CascadeClassifier exists 1:1 in C++ for the webOS port.
"""
import cv2


class FaceMasker:
    def __init__(self, cfg):
        f = cfg.get("face", {})
        self.enabled = f.get("exclude", True)
        self.pad_side = f.get("padSide", 0.25)    # ears
        self.pad_top = f.get("padTop", 0.5)       # hair
        self.pad_bottom = f.get("padBottom", 0.4)  # chin / beard / neck
        self.hold_frames = f.get("holdFrames", 10)
        # OpenCV 5 contrib (our venv) has CascadeClassifier but doesn't ship the
        # XML data file, so prefer models/ (tools/get_models.py downloads it).
        from pathlib import Path
        name = "haarcascade_frontalface_default.xml"
        candidates = [Path(__file__).resolve().parent.parent / "models" / name]
        if hasattr(cv2, "data"):
            candidates.append(Path(cv2.data.haarcascades) / name)
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise RuntimeError(f"Face cascade {name} not found - run tools/get_models.py")
        self.cascade = cv2.CascadeClassifier(str(path))
        if self.cascade.empty():
            raise RuntimeError(f"Could not load face cascade: {path}")
        self.faces = []
        self.hold = 0

    def update(self, frame):
        """Returns a list of padded face rects (x, y, w, h) in frame coords."""
        if not self.enabled:
            return []
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = self.cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(40, 40))
        if len(found):
            self.faces = [self._pad(f, frame.shape) for f in found]
            self.hold = self.hold_frames
        elif self.hold:
            # Detector flickers frame to frame (head turned, motion blur):
            # keep the last face for a few frames instead of dropping it.
            self.hold -= 1
        else:
            self.faces = []
        return self.faces

    def _pad(self, rect, shape):
        x, y, w, h = rect
        H, W = shape[:2]
        x0 = max(0, int(x - w * self.pad_side))
        y0 = max(0, int(y - h * self.pad_top))
        x1 = min(W, int(x + w * (1 + self.pad_side)))
        y1 = min(H, int(y + h * (1 + self.pad_bottom)))
        return (x0, y0, x1 - x0, y1 - y0)


def rects_to_roi(rects, roi_x, roi_y):
    """Shift frame-coordinate rects into ROI coordinates."""
    return [(x - roi_x, y - roi_y, w, h) for x, y, w, h in rects]
