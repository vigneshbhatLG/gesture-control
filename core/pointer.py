"""Pointer mode: Magic-Remote-style air mouse. Pure logic (no OS calls) so it
ports 1:1 to the TV service; core/mouse.py is the laptop output.

  OFF     -> hold ☝️ (index up, others curled) for activateMs -> ACTIVE
  ACTIVE  -> cursor follows the INDEX KNUCKLE (landmark 5), relative motion
             (like a mouse: starts where the cursor already is)
          -> thumb tap = click: thumb gap drops below pressGap after having
             been above releaseGap (hysteresis, so jitter can't double-click)
          -> cursor moves ONLY while the hand is pointing (thumb is ignored by
             the pointing rule, so a thumb-tap click keeps pointing)
          -> ✌️ (index + middle up) = SCROLL sub-mode, joystick style: where ✌️
             starts is the centre; move the hand away -> page keeps scrolling
             that way, further = faster, back to centre = stop. Deadzone +
             axis lock so a slightly diagonal hand scrolls only one way.
             ☝️ again -> back to moving the cursor (re-anchored, no jump).
          -> ends on: neither ☝️ nor ✌️ for notPointingExitMs (open palm, ...),
             hand lost for exitMs, or fist held for exitMs

Why the knuckle: the fingertip moves when you tap the thumb; the knuckle barely
does. Cursor is also frozen while the thumb is pressed, so clicks land where
you aimed.

Smoothing: One Euro filter (Casiez et al. 2012) - heavy smoothing when the hand
is still (kills jitter), light smoothing when it moves fast (no lag).
"""
import math
from collections import deque
from statistics import median

from core.gestures import palm_length_raw, thumb_gap

OFF, ACTIVE = "OFF", "ACTIVE"
MOVE, SCROLL = "MOVE", "SCROLL"  # sub-modes while ACTIVE: ☝️ = MOVE, ✌️ = SCROLL
ANCHOR = 5  # index finger knuckle (MCP)
WHEEL = 120  # Windows: one wheel "notch" = 120 units (smaller = smooth scroll)


class OneEuro:
    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


class Pointer:
    def __init__(self, cfg, screen_w, screen_h, aspect, get_cursor=None):
        self.screen = (screen_w, screen_h)
        self.aspect = aspect          # camera frame width / height
        self.get_cursor = get_cursor  # returns current OS cursor (x, y) or None
        self.configure(cfg)
        self.state = OFF
        self.point_since = None
        self.last_seen = 0.0
        self.fist_since = None
        self.cursor = (screen_w / 2, screen_h / 2)
        self.pressed = True
        self.last_click = -1e9
        self.gap = None       # smoothed (median) thumb gap
        self.gap_raw = None   # this frame's raw gap (for logging/calibration)
        self.gap_hist = deque(maxlen=5)
        self.below = 0        # consecutive frames with gap under pressGap
        self.click_held = False  # a click fired and the thumb is still pressed
        self.mode = MOVE
        self.scroll_offset = (0.0, 0.0)  # joystick offset in palm lengths (UI)
        self.scroll_vel = (0.0, 0.0)     # notches/s (UI)

    def configure(self, cfg):
        p = cfg["pointer"]
        self.enabled = p["enabled"]
        self.activate_ms = p["activateMs"]
        self.exit_ms = p["exitMs"]
        self.press_gap = p["pressGap"]
        self.release_gap = p["releaseGap"]
        self.debounce = p["clickDebounceMs"]
        self.press_frames = p.get("pressFrames", 2)
        self.not_point_exit = p.get("notPointingExitMs", 300)
        self.require_armed = p.get("requireArmed", True)
        # Adaptive click thresholds: live data showed the user's "thumb out" level
        # drifting between sessions (0.70-0.85 vs 0.55-0.75), which broke fixed ones.
        a = p.get("adaptive", {})
        self.adaptive = a.get("enabled", True)
        self.press_ratio = a.get("pressRatio", 0.5)
        self.release_ratio = a.get("releaseRatio", 0.75)
        self.open_floor = a.get("openFloor", 0.5)
        self.open_pct = a.get("openPercentile", 85)
        self.open_hist = deque(maxlen=a.get("windowFrames", 90))  # ~3 s at 30 FPS
        self.fixed_press, self.fixed_release = self.press_gap, self.release_gap
        self.open_ref = None
        # gestureSensitivity 1..5 -> how far the hand must move to cross the screen
        self.gain = 1.0 + 0.5 * cfg.get("gestureSensitivity", 3)
        self.fx = OneEuro(p["minCutoff"], p["beta"])
        self.fy = OneEuro(p["minCutoff"], p["beta"])
        s = p.get("scroll", {})
        self.scroll_enabled = s.get("enabled", True)
        self.deadzone = s.get("deadzone", 0.15)      # palm lengths
        self.max_offset = s.get("maxOffset", 0.8)    # palm lengths -> full speed
        self.max_nps = s.get("maxNotchesPerSec", 12)
        self.axis_lock = s.get("axisLock", True)
        # joystick works in palm-length units -> beta scaled down accordingly
        self.sfx = OneEuro(s.get("minCutoff", 1.5), s.get("beta", 0.5))
        self.sfy = OneEuro(s.get("minCutoff", 1.5), s.get("beta", 0.5))

    @property
    def active(self):
        return self.state == ACTIVE

    def _anchor_px(self, raw_points):
        """Knuckle position in 'screen pixel' units (physical-aspect corrected:
        the same hand distance moves the cursor the same in x and y)."""
        x, y = raw_points[ANCHOR]
        h = self.screen[1]
        return x * self.aspect * self.gain * h, y * self.gain * h

    def _start(self, raw_points, now):
        self.state = ACTIVE
        self.last_seen = now
        start = self.get_cursor() if self.get_cursor else None
        self.cursor = start or (self.screen[0] / 2, self.screen[1] / 2)
        self.fx.reset()
        self.fy.reset()
        ax, ay = self._anchor_px(raw_points)
        self.fx(ax, now / 1000)
        self.fy(ay, now / 1000)
        self.prev_anchor = (ax, ay)
        self.pressed = True  # thumb must open (release) before the first click
        self.fist_since = None
        self.not_point_since = None
        self.resync = False
        self.gap_hist.clear()
        self.open_hist.clear()
        self.click_held = False
        self.below = 0
        self.mode = MOVE

    def _enter_scroll(self, raw_points, now):
        self.mode = SCROLL
        x, y = raw_points[ANCHOR]
        self.scroll_scale = max(palm_length_raw(raw_points, self.aspect), 1e-3)
        self.sfx.reset()
        self.sfy.reset()
        self.scroll_center = (self.sfx(x * self.aspect / self.scroll_scale, now / 1000),
                              self.sfy(y / self.scroll_scale, now / 1000))
        self.scroll_t = now
        self.acc = [0.0, 0.0]
        self.pressed = True  # back in MOVE, the thumb must open before a click

    def _speed(self, o):
        a = abs(o)
        if a < self.deadzone:
            return 0.0
        frac = min((a - self.deadzone) / max(self.max_offset - self.deadzone, 1e-6), 1.0)
        return math.copysign(frac * self.max_nps, o)

    def _scroll(self, raw_points, now):
        """Joystick: offset from the centre -> scroll speed. Returns wheel units
        (wx, wy): wy > 0 = scroll up, wx > 0 = scroll right."""
        x, y = raw_points[ANCHOR]
        fx = self.sfx(x * self.aspect / self.scroll_scale, now / 1000)
        fy = self.sfy(y / self.scroll_scale, now / 1000)
        ox, oy = fx - self.scroll_center[0], fy - self.scroll_center[1]  # +y = hand down
        vx, vy = self._speed(ox), self._speed(oy)
        if self.axis_lock:
            if abs(ox) >= abs(oy):
                vy = 0.0
            else:
                vx = 0.0
        dt = max(now - self.scroll_t, 0.0) / 1000
        self.scroll_t = now
        self.acc[0] += vx * dt * WHEEL
        self.acc[1] += -vy * dt * WHEEL  # hand up (oy < 0) -> scroll up (wheel +)
        out = (int(self.acc[0]), int(self.acc[1]))
        self.acc[0] -= out[0]
        self.acc[1] -= out[1]
        self.scroll_offset, self.scroll_vel = (ox, oy), (vx, -vy)
        return out

    def _adapt(self):
        """Learn the current 'thumb out' level and derive press/release from it.
        Not learned while pressed, so holding a click can't drag thresholds down."""
        if not self.adaptive:
            self.press_gap, self.release_gap = self.fixed_press, self.fixed_release
            return
        # Freeze learning only while a real click is held. (Freezing on `pressed`
        # deadlocked: pointer starts 'pressed' until the thumb opens, so a thumb
        # that never passed the initial release mark was never learned.)
        if not self.click_held:
            self.open_hist.append(self.gap)
        if len(self.open_hist) >= 10:
            vals = sorted(self.open_hist)
            ref = vals[min(len(vals) - 1, int(len(vals) * self.open_pct / 100))]
            self.open_ref = max(self.open_floor, ref)
        else:
            self.open_ref = max(self.open_floor, self.fixed_release / self.release_ratio)
        self.press_gap = self.open_ref * self.press_ratio
        self.release_gap = self.open_ref * self.release_ratio

    def stop(self):
        self.state = OFF
        self.point_since = None

    def update(self, hand, now, pointing=False, fist=False, armed=True,
               two_fingers=False):
        """hand: {"raw": 21 (x, y) image coords, "norm": normalised} or None.
        armed: gesture control is armed (open palm). With requireArmed, ☝️/✌️
        only start the pointer when armed - same rule as every other gesture.
        Returns events: {"started", "stopped", "moved", "click", "scroll",
        "reason"}; scroll = (wx, wy) wheel units this frame."""
        ev = {"started": False, "stopped": False, "moved": False, "click": False,
              "scroll": (0, 0), "reason": None}
        two_fingers = two_fingers and self.scroll_enabled
        if not self.enabled:
            if self.active:
                self.stop()
                ev["stopped"], ev["reason"] = True, "disabled"
            return ev

        if self.state == OFF:
            if (hand is not None and (pointing or two_fingers)
                    and (armed or not self.require_armed)):
                if self.point_since is None:
                    self.point_since = now
                if now - self.point_since >= self.activate_ms:
                    self._start(hand["raw"], now)
                    ev["started"] = True
                    if two_fingers:  # ✌️ straight away = start in scroll mode
                        self._enter_scroll(hand["raw"], now)
            else:
                self.point_since = None
            return ev

        # ACTIVE
        if hand is None:
            if now - self.last_seen > self.exit_ms:
                self.stop()
                ev["stopped"], ev["reason"] = True, "lost"
            return ev
        self.last_seen = now

        if fist:
            self.fist_since = self.fist_since or now
            if now - self.fist_since >= self.exit_ms:
                self.stop()
                ev["stopped"], ev["reason"] = True, "fist"
                return ev
        else:
            self.fist_since = None

        # Only a pointing hand drives the cursor (thumb is free, so clicks don't
        # break this). Any other shape (open palm, 🤘, ...) freezes the cursor and,
        # if it lasts notPointingExitMs, exits pointer mode. (Live bug 2026-11-13:
        # cursor kept moving with an open palm.)
        if not (pointing or two_fingers):
            self.not_point_since = self.not_point_since or now
            if now - self.not_point_since >= self.not_point_exit:
                self.stop()
                ev["stopped"], ev["reason"] = True, "shape"
            self.resync = True
            return ev
        self.not_point_since = None

        # ✌️ -> scroll (joystick). Cursor frozen, no clicks.
        if two_fingers:
            if self.mode != SCROLL:
                self._enter_scroll(hand["raw"], now)
            ev["scroll"] = self._scroll(hand["raw"], now)
            self.resync = True  # re-anchor the cursor when back to ☝️
            return ev
        if self.mode == SCROLL:
            self.mode = MOVE  # ☝️ again (resync below re-anchors the cursor)
        if self.resync:
            # back to pointing after a short break: re-anchor so the cursor
            # doesn't jump by however far the hand moved meanwhile
            ax, ay = self._anchor_px(hand["raw"])
            self.fx.reset()
            self.fy.reset()
            self.prev_anchor = (self.fx(ax, now / 1000), self.fy(ay, now / 1000))
            self.resync = False

        # Click: thumb tap. Raw thumb landmark jitters a lot in the pointing pose
        # (live test: ~85 false clicks), so: median of last 5 frames, press must
        # last pressFrames frames, hysteresis between press/release, debounce.
        self.gap_raw = thumb_gap(hand["norm"])
        self.gap_hist.append(self.gap_raw)
        self.gap = median(self.gap_hist)
        self._adapt()
        if self.pressed:
            if self.gap > self.release_gap:
                self.pressed = False
                self.click_held = False
            self.below = 0
        else:
            self.below = self.below + 1 if self.gap < self.press_gap else 0
            if self.below >= self.press_frames and now - self.last_click >= self.debounce:
                self.pressed = True
                self.click_held = True
                self.below = 0
                self.last_click = now
                ev["click"] = True

        # Move (relative, filtered). Frozen while the thumb is pressed.
        ax, ay = self._anchor_px(hand["raw"])
        fx, fy = self.fx(ax, now / 1000), self.fy(ay, now / 1000)
        if not self.pressed:
            dx, dy = fx - self.prev_anchor[0], fy - self.prev_anchor[1]
            w, h = self.screen
            nx = min(max(self.cursor[0] + dx, 0), w - 1)
            ny = min(max(self.cursor[1] + dy, 0), h - 1)
            ev["moved"] = (nx, ny) != self.cursor
            self.cursor = (nx, ny)
        self.prev_anchor = (fx, fy)
        return ev
