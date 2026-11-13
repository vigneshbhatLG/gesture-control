"""Custom pose gestures: few-shot template matching (no neural-net training).

Each gesture = a name + 5 samples of the 21 hand landmarks, normalised so
position in the frame and distance from the camera don't matter:
  - wrist moved to (0, 0)
  - scaled so wrist -> middle-finger knuckle = 1 ("palm length")
  - x corrected for frame aspect ratio (landmarks come normalised per axis)
  - left hand flipped to look like a right hand (optional: either hand works)
Rotation is NOT normalised on purpose: thumbs-up vs thumbs-down must differ.

Distance between two hands = RMS of the 21 point distances, in palm lengths.
RMS (not mean) so one finger that differs a lot still counts strongly.

Storage: x,y only = 42 numbers per sample (TV: float16 -> ~84 bytes).
"""
import json
import uuid
from itertools import combinations
from pathlib import Path

import numpy as np

WRIST, MIDDLE_MCP = 0, 9


def normalize(points, aspect, label="Right", mirror_left=True):
    """points: 21 (x, y) in 0..1 image coords. aspect: frame width / height.
    Returns (21, 2) float32 array, or None if the hand is degenerate."""
    p = np.asarray(points, np.float32).copy()
    p[:, 0] *= aspect  # make x and y the same units
    p -= p[WRIST]
    scale = float(np.linalg.norm(p[MIDDLE_MCP]))
    if scale < 1e-6:
        return None
    p /= scale
    if mirror_left and label == "Left":
        p[:, 0] = -p[:, 0]
    return p


TIPS, PIPS = (8, 12, 16, 20), (6, 10, 14, 18)  # index..pinky
THUMB_TIP, INDEX_MCP = 4, 5


def is_open_palm(p):
    """Built-in activation pose: all 5 fingers extended. p = normalised hand.
    Rule-based (no template), works for anyone, rotation independent:
      finger extended = tip further from wrist than its middle joint (x1.15)
      thumb extended  = thumb tip > 0.5 palm lengths from index knuckle"""
    d = np.linalg.norm(p, axis=1)  # distance of every point from the wrist
    fingers = all(d[t] > d[j] * 1.15 for t, j in zip(TIPS, PIPS))
    thumb = float(np.linalg.norm(p[THUMB_TIP] - p[INDEX_MCP])) > 0.5
    return fingers and thumb


PALM_POINTS = (5, 6, 9, 10, 13)  # index/middle/ring knuckles + middle joints


def is_fist(p):
    """Built-in disarm pose: closed fist. p = normalised hand.
      finger curled = tip CLOSER to wrist than its middle joint
      thumb tucked  = thumb tip within 0.4 palm lengths of the knuckles/fingers
    The thumb check keeps thumbs-up / thumbs-down from counting as a fist."""
    d = np.linalg.norm(p, axis=1)
    fingers = all(d[t] < d[j] for t, j in zip(TIPS, PIPS))
    thumb = min(float(np.linalg.norm(p[THUMB_TIP] - p[j])) for j in PALM_POINTS) < 0.4
    return fingers and thumb


def is_pointing(p):
    """Built-in pointer pose ☝️: index extended, middle/ring/pinky curled.
    Thumb is ignored here - it's used for the click (thumb tap)."""
    d = np.linalg.norm(p, axis=1)
    index_up = d[8] > d[6] * 1.15
    others_curled = all(d[t] < d[j] for t, j in zip(TIPS[1:], PIPS[1:]))
    return index_up and others_curled


def is_two_fingers(p):
    """Built-in scroll pose ✌️: index + middle extended, ring + pinky curled.
    Thumb ignored (same as pointing)."""
    d = np.linalg.norm(p, axis=1)
    up = all(d[t] > d[j] * 1.15 for t, j in ((8, 6), (12, 10)))
    curled = all(d[t] < d[j] for t, j in ((16, 14), (20, 18)))
    return up and curled


def palm_length_raw(raw_points, aspect):
    """Wrist -> middle knuckle in image units (x aspect-corrected). Used to make
    the scroll joystick independent of distance from the camera."""
    (x0, y0), (x9, y9) = raw_points[WRIST], raw_points[MIDDLE_MCP]
    return float(np.hypot((x9 - x0) * aspect, y9 - y0))


def thumb_gap(p):
    """Debug/calibration: thumb tip distance to nearest knuckle/finger point."""
    return min(float(np.linalg.norm(p[THUMB_TIP] - p[j])) for j in PALM_POINTS)


def distance(a, b):
    """RMS distance between two normalised hands, in palm lengths."""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


class GestureStore:
    def __init__(self, cfg, root):
        g = cfg["gestures"]
        self.path = Path(root) / g["file"]
        self.match_threshold = g["matchThreshold"]
        self.consistency_max = g["consistencyMax"]
        self.distinct_min = g["distinctMin"]
        self.gestures = []
        self.load()

    # ---------- persistence ----------
    def load(self):
        if not self.path.exists():
            self.gestures = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for g in data["gestures"]:
            g["samples"] = [np.array(s, np.float32).reshape(21, 2)
                            for s in g["samples"]]
        self.gestures = data["gestures"]

    def save(self):
        out = {"version": 1, "gestures": [
            {**g, "samples": [np.round(s.reshape(-1), 3).tolist()
                              for s in g["samples"]]}
            for g in self.gestures]}
        self.path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---------- recording ----------
    def check_new(self, samples):
        """Returns (ok, message). Runs the consistency + distinctiveness checks."""
        if sum(is_open_palm(s) for s in samples) > len(samples) // 2:
            return False, ("Open palm is reserved for arming gesture control. "
                           "Use a different gesture.")
        if sum(is_fist(s) for s in samples) > len(samples) // 2:
            return False, ("Closed fist is reserved for disarming gesture control. "
                           "Use a different gesture.")
        if sum(is_pointing(s) for s in samples) > len(samples) // 2:
            return False, ("Pointing finger is reserved for pointer mode. "
                           "Use a different gesture.")
        if sum(is_two_fingers(s) for s in samples) > len(samples) // 2:
            return False, ("Two fingers (V) is reserved for scrolling. "
                           "Use a different gesture.")
        pair = [distance(a, b) for a, b in combinations(samples, 2)]
        spread = float(np.mean(pair))
        if spread > self.consistency_max:
            return False, (f"Samples not consistent (spread {spread:.2f} > "
                           f"{self.consistency_max}). Keep the gesture the same.")
        for g in self.gestures:
            d = min(distance(a, b) for a in samples for b in g["samples"])
            if d < self.distinct_min:
                return False, (f'Too similar to "{g["name"]}" (distance {d:.2f} '
                               f"< {self.distinct_min}). Use a different gesture.")
        return True, f"OK (spread {spread:.2f})"

    def add(self, name, samples):
        # No action stored here: bindings live in config gestureMap (id -> action),
        # i.e. in settingsservice on the TV. Templates and bindings stay separate.
        g = {"id": "g_" + uuid.uuid4().hex[:6], "name": name, "type": "pose",
             "samples": list(samples)}
        self.gestures.append(g)
        self.save()
        return g

    def remove(self, gesture_id):
        before = len(self.gestures)
        self.gestures = [g for g in self.gestures if g["id"] != gesture_id]
        if len(self.gestures) != before:
            self.save()
            return True
        return False

    def remove_last(self):
        if self.gestures:
            g = self.gestures.pop()
            self.save()
            return g
        return None

    # ---------- matching ----------
    def match(self, hand):
        """Returns (best_gesture or None, best_distance, runner_up_distance).
        best_gesture is None if nothing is within matchThreshold."""
        scored = sorted(
            (min(distance(hand, s) for s in g["samples"]), i)
            for i, g in enumerate(self.gestures))
        if not scored:
            return None, None, None
        best_d, best_i = scored[0]
        runner = scored[1][0] if len(scored) > 1 else None
        best = self.gestures[best_i] if best_d <= self.match_threshold else None
        return best, best_d, runner
