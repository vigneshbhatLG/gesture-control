"""MediaPipe hand landmarks (pretrained; we train nothing).

Wraps the HandLandmarker task in VIDEO mode, which tracks the hand between
frames (the palm detector only runs when the hand is lost) - the same
detect -> track idea we want on the TV.

Model is lazy-loaded on first use and can be unloaded to free RAM.
On the TV this becomes: the two .tflite files run directly via TFLite in C++.
"""
import time

import cv2
import mediapipe as mp

vision = mp.tasks.vision
CONNECTIONS = [(c.start, c.end)
               for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


class HandTracker:
    def __init__(self, cfg, root):
        h = cfg["hands"]
        self.model_path = str(root / h["model"])
        self.max_hands = h["maxHands"]
        self.det_conf = h["minDetectionConf"]
        self.presence_conf = h["minPresenceConf"]
        self.track_conf = h["minTrackingConf"]
        # MediaPipe's Left/Right comes out swapped with our mirrored frames
        # (verified live 2026-11-13). Config switch, because a TV camera
        # may deliver unmirrored frames and need it off.
        self.swap_handedness = h.get("swapHandedness", False)
        self.landmarker = None
        self.load_ms = None
        self.last_ms = 0.0

    @property
    def loaded(self):
        return self.landmarker is not None

    def load(self):
        t = time.perf_counter()
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.det_conf,
            min_hand_presence_confidence=self.presence_conf,
            min_tracking_confidence=self.track_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.load_ms = (time.perf_counter() - t) * 1000

    def unload(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def detect(self, frame_bgr, timestamp_ms):
        """Returns a list of hands: {points: [(x, y)] * 21 normalised 0..1,
        label: "Left"/"Right", score}."""
        if not self.loaded:
            self.load()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t = time.perf_counter()
        result = self.landmarker.detect_for_video(image, int(timestamp_ms))
        self.last_ms = (time.perf_counter() - t) * 1000

        hands = []
        for i, lms in enumerate(result.hand_landmarks):
            cat = result.handedness[i][0] if result.handedness else None
            label = cat.category_name if cat else "?"
            if self.swap_handedness:
                label = {"Left": "Right", "Right": "Left"}.get(label, label)
            hands.append({
                "points": [(lm.x, lm.y) for lm in lms],
                "label": label,
                "score": cat.score if cat else 0.0,
            })
        return hands
