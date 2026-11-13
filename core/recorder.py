"""Record-mode UI state machine (no drawing here, just logic).

  r       -> NAMING: type a name in the window, Enter = confirm, Esc = cancel
  Enter   -> CAPTURE: one sample every sampleIntervalMs while a hand is visible
  done    -> run checks, save or explain why not
"""
import re

OFF, NAMING, CAPTURE = "off", "naming", "capture"


class Recorder:
    def __init__(self, cfg, store):
        g = cfg["gestures"]
        self.n = g["samplesPerGesture"]
        self.interval = g["sampleIntervalMs"]
        self.store = store
        self.state = OFF
        self.name = ""
        self.samples = []
        self.next_at = 0.0
        self.message = ""
        self.message_until = 0.0

    @property
    def active(self):
        return self.state != OFF

    def say(self, text, now_ms, ms=4000):
        self.message, self.message_until = text, now_ms + ms

    def start(self):
        self.state, self.name, self.samples = NAMING, "", []

    def key(self, k, now_ms):
        """Handle a key while recording. Returns True if consumed."""
        if self.state == NAMING:
            if k == 27:  # Esc
                self.state = OFF
                self.say("Recording cancelled", now_ms)
            elif k in (10, 13):  # Enter (CR on some systems, LF on others)
                if self.name.strip():
                    self.state = CAPTURE
                    self.next_at = now_ms + 2000  # time to get into position
                else:
                    self.say("Type a name first", now_ms, 2000)
            elif k == 8:  # Backspace
                self.name = self.name[:-1]
            elif 32 <= k < 127 and len(self.name) < 20:
                ch = chr(k)
                if re.match(r"[A-Za-z0-9 _-]", ch):
                    self.name += ch
            return True
        if self.state == CAPTURE and k == 27:
            self.state = OFF
            self.say("Recording cancelled", now_ms)
            return True
        return False

    def update(self, hand_norm, now_ms):
        """Call every frame during CAPTURE with the normalised hand (or None)."""
        if self.state != CAPTURE or now_ms < self.next_at or hand_norm is None:
            return
        self.samples.append(hand_norm)
        self.next_at = now_ms + self.interval
        if len(self.samples) >= self.n:
            ok, msg = self.store.check_new(self.samples)
            if ok:
                self.store.add(self.name.strip(), self.samples)
                self.say(f'Saved "{self.name.strip()}" - {msg}', now_ms)
            else:
                self.say(f"Not saved: {msg}", now_ms, 6000)
            self.state = OFF

    def lines(self, now_ms, hand_visible):
        """Text to show on screen."""
        if self.state == NAMING:
            hint = self.message if now_ms < self.message_until else "Enter = start   Esc = cancel"
            return [f"Gesture name: {self.name}_", hint]
        if self.state == CAPTURE:
            left = max(0.0, (self.next_at - now_ms) / 1000)
            status = f"in {left:.1f}s" if hand_visible else "show your hand!"
            return [f'Recording "{self.name}"  sample {len(self.samples) + 1}/{self.n}  {status}',
                    "Hold the gesture, move it slightly between samples. Esc = cancel"]
        if now_ms < self.message_until:
            return [self.message]
        return []
