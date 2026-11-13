"""Turns per-frame matches into safe, deliberate actions.

  DISARMED  -> hold open palm for armHoldMs            -> ARMED
  ARMED     -> hold a custom gesture for holdMs        -> FIRE -> COOLDOWN
  COOLDOWN  -> after gestureCooldownMs                 -> ARMED
  ARMED     -> hold closed fist for disarmHoldMs       -> DISARMED
  ARMED     -> no gesture fired for armTimeoutMs       -> DISARMED (safety net)

Why:
  - arming: random hand movement in front of the TV does nothing
  - hold:   hand shapes passing through on the way to another gesture don't fire
  - cooldown + release: one thumbs-up = one action (unless repeatWhileHeld)
  - dropout tolerance: a 1-frame tracking flicker doesn't reset the hold timer

Pure logic (no camera, no drawing) -> ports 1:1 to the TV service.
"""

DISARMED, ARMED, COOLDOWN = "DISARMED", "ARMED", "COOLDOWN"


class Trigger:
    def __init__(self, cfg, repeat_for=None):
        """repeat_for(gesture) -> bool: per-gesture repeat (e.g. from its bound
        action). Falls back to trigger.repeatWhileHeld if not given."""
        self.repeat_for = repeat_for
        self.configure(cfg)
        self.state = DISARMED
        self.cand = None        # gesture id (or "palm") being held
        self.cand_since = 0.0
        self.cand_seen = 0.0
        self.armed_until = 0.0
        self.cooldown_until = 0.0
        self.block = None       # gesture that must be released before re-firing
        self.last_fired = None  # (gesture, time) for the UI
        self._gesture = None    # dict of the gesture being held

    def configure(self, cfg):
        """(Re)load timing settings - called again on live config reload."""
        t = cfg["trigger"]
        self.arm_hold = t["armHoldMs"]
        self.arm_timeout = t["armTimeoutMs"]
        self.disarm_hold = t["disarmHoldMs"]
        self.hold = t["holdMs"]
        self.dropout = t["dropoutMs"]
        self.repeat = t["repeatWhileHeld"]
        self.repeat_ms = t["repeatMs"]
        self.cooldown = cfg["gestureCooldownMs"]  # settingsservice key

    def _track(self, key, now):
        """Update the 'currently held' candidate with dropout tolerance."""
        if key is not None:
            if key != self.cand:
                self.cand, self.cand_since = key, now
            self.cand_seen = now
        elif self.cand is not None and now - self.cand_seen > self.dropout:
            self.cand = None
        if self.block is not None and self.cand != self.block:
            self.block = None  # released

    @property
    def armed(self):
        return self.state != DISARMED

    def keep_alive(self, now):
        """Pointer mode in use counts as activity: don't time out while pointing."""
        if self.armed:
            self.armed_until = now + self.arm_timeout

    def disarm(self):
        self.state = DISARMED
        self.cand = None

    def held_ms(self, now):
        return now - self.cand_since if self.cand is not None else 0.0

    def update(self, gesture, palm, now, fist=False):
        """gesture: matched gesture dict or None. palm / fist: built-in poses
        (open palm = arm, closed fist = disarm). Returns fired gesture or None."""
        key = "palm" if palm else "fist" if fist else (gesture["id"] if gesture else None)
        self._track(key, now)

        # Closed fist held -> disarm right away (from ARMED or COOLDOWN)
        if (self.state != DISARMED and self.cand == "fist"
                and self.held_ms(now) >= self.disarm_hold):
            self.state = DISARMED
            self.block = "fist"
            return None
        if gesture is not None:
            self._gesture = gesture  # remember dict for the id being held

        if self.state == DISARMED:
            if self.cand == "palm" and self.held_ms(now) >= self.arm_hold:
                self.state = ARMED
                self.armed_until = now + self.arm_timeout
                self.block = "palm"
            return None

        if self.state == COOLDOWN:
            if now < self.cooldown_until:
                return None
            self.state = ARMED

        # ARMED
        if self.cand == "palm":
            self.armed_until = now + self.arm_timeout  # palm keeps it armed
            return None
        if now > self.armed_until:
            self.state = DISARMED
            self.cand = None
            return None
        if (self.cand is not None and self.cand not in (self.block, "fist")
                and self.held_ms(now) >= self.hold):
            fired = self._gesture
            self.state = COOLDOWN
            self.armed_until = now + self.arm_timeout
            self.last_fired = (fired, now)
            rep = self.repeat_for(fired) if self.repeat_for else self.repeat
            if rep:
                # still held -> fire again every repeatMs (no new hold needed)
                self.cooldown_until = now + self.repeat_ms
                self.cand_since = now - self.hold
            else:
                self.cooldown_until = now + self.cooldown
                self.block = self.cand  # must release first
            return fired
        return None

    def progress(self, now):
        """0..1 progress of the current hold (for a progress bar), or None."""
        if self.cand is None or self.cand == self.block:
            return None
        if self.state == DISARMED and self.cand == "palm":
            return min(1.0, self.held_ms(now) / self.arm_hold)
        if self.state != DISARMED and self.cand == "fist":
            return min(1.0, self.held_ms(now) / self.disarm_hold)
        if self.state == ARMED and self.cand != "palm":
            return min(1.0, self.held_ms(now) / self.hold)
        return None
