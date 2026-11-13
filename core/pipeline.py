"""Lean pipeline: only run the model when it can matter.

  IDLE   - no model calls. Cheap motion check on a tiny 64x48 gray frame.
  SEARCH - motion seen, no hand yet. Model runs at a LOW rate (searchFps).
  TRACK  - hand found. Model runs every frame.

No hand for idleAfterMs  -> back to IDLE.
IDLE for unloadAfterMs   -> model unloaded (frees RAM).

Ports directly to the TV service (motion check = plain C++ on the
camera's low-res luma; model calls = TFLite).
"""
import cv2
import numpy as np

IDLE, SEARCH, TRACK = "IDLE", "SEARCH", "TRACK"


class MotionGate:
    def __init__(self, cfg):
        m = cfg["motion"]
        self.size = tuple(m["size"])
        self.thresh = m["threshold"]
        self.min_ratio = m["minChangedRatio"]
        self.prev = None

    def moved(self, frame):
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.size,
                           interpolation=cv2.INTER_AREA)
        if self.prev is None:
            self.prev = small
            return False
        diff = cv2.absdiff(small, self.prev)
        self.prev = small
        return np.count_nonzero(diff > self.thresh) / diff.size >= self.min_ratio


class Pipeline:
    def __init__(self, cfg, tracker):
        p = cfg["pipeline"]
        self.search_interval = 1000.0 / p["searchFps"]
        self.idle_after = p["idleAfterMs"]
        self.unload_after = p["unloadAfterMs"]
        self.tracker = tracker
        self.gate = MotionGate(cfg)
        self.state = IDLE
        self.last_hand = 0.0
        self.last_search = -1e9
        self.idle_since = 0.0
        self.model_calls = 0  # for "model runs per second" stat

    def step(self, frame, now_ms):
        moved = self.gate.moved(frame)

        if self.state == IDLE:
            if not moved:
                if (self.tracker.loaded
                        and now_ms - self.idle_since > self.unload_after):
                    self.tracker.unload()
                return []
            self.state = SEARCH
            self.last_hand = now_ms  # start the idle timer from wake-up

        if self.state == SEARCH and now_ms - self.last_search < self.search_interval:
            hands = []  # throttled: skip the model this frame
        else:
            if self.state == SEARCH:
                self.last_search = now_ms
            hands = self.tracker.detect(frame, now_ms)
            self.model_calls += 1

        if hands:
            self.state = TRACK
            self.last_hand = now_ms
        else:
            if self.state == TRACK:
                self.state = SEARCH
            if now_ms - self.last_hand > self.idle_after:
                self.state = IDLE
                self.idle_since = now_ms
        return hands
