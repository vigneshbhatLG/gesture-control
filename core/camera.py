"""Camera source abstraction.

The rest of the code only calls read() and release(). On the TV this class
gets replaced by one backed by com.webos.service.camera (built-in or USB cam),
and nothing else has to change.
"""
import cv2

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


class Camera:
    def __init__(self, cfg):
        c = cfg["camera"]
        backend = BACKENDS[c.get("backend", "any")]
        self.cap = cv2.VideoCapture(c["index"], backend)
        self.size = (c["width"], c["height"])
        self.mirror = c.get("mirror", True)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera (in use by Teams/Zoom? "
                               "run tools/probe_cameras.py to find the right index)")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        # Drivers often ignore the requested size, so enforce it here.
        # Small frames = fast processing (matters a lot on a TV SoC).
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        self.cap.release()
