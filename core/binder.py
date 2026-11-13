"""Bind-menu state machine (GTA-style remap), no drawing here.

  b (while a gesture is matched) -> menu of actions
  1..N bind, 0 unbind, Esc cancel

Rule: one action per gesture (dict key). If the chosen action was bound to
another gesture, that binding MOVES to this gesture (like a game's key swap).
"""
from core.actions import ORDER, label


class Binder:
    def __init__(self):
        self.gesture = None
        self.message = ""
        self.message_until = 0.0

    @property
    def active(self):
        return self.gesture is not None

    def start(self, gesture):
        self.gesture = gesture

    def say(self, text, now_ms, ms=3500):
        self.message, self.message_until = text, now_ms + ms

    def key(self, k, now_ms, gesture_map):
        """Handle a key while the menu is open. Mutates gesture_map.
        Returns (consumed, changed)."""
        if not self.active:
            return False, False
        gid, name = self.gesture["id"], self.gesture["name"]
        if k == 27:
            self.gesture = None
            self.say("Bind cancelled", now_ms)
            return True, False
        if k == ord("0"):
            gesture_map.pop(gid, None)
            self.gesture = None
            self.say(f'"{name}" unbound', now_ms)
            return True, True
        if ord("1") <= k <= ord("9") and k - ord("1") < len(ORDER):
            action = ORDER[k - ord("1")]
            moved = [g for g, a in gesture_map.items() if a == action and g != gid]
            for g in moved:
                del gesture_map[g]
            gesture_map[gid] = action
            self.gesture = None
            extra = " (moved from another gesture)" if moved else ""
            self.say(f'"{name}" -> {label(action)}{extra}', now_ms)
            return True, True
        return True, False  # swallow other keys while menu is open

    def lines(self, now_ms, gesture_map):
        if self.active:
            cur = label(gesture_map.get(self.gesture["id"]))
            out = [f'Bind "{self.gesture["name"]}" (now: {cur}) - press a number:']
            row = "   ".join(f"{i + 1} {label(a)}" for i, a in enumerate(ORDER))
            out.append(row)
            out.append("0 unbind   Esc cancel")
            return out
        if now_ms < self.message_until:
            return [self.message]
        return []
