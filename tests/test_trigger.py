"""Timeline tests for the arm / hold / cooldown trigger (33 ms frames)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import copy

from core.config import load_config
from core.trigger import ARMED, DISARMED, Trigger

UP = {"id": "g_up", "name": "volume up", "action": None}
YT = {"id": "g_yt", "name": "open youtube", "action": None}
fails = 0


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print("PASS" if cond else "FAIL", name, extra)


class Sim:
    def __init__(self, cfg):
        self.t, self.tr, self.fired = 0.0, Trigger(cfg), []

    def run(self, ms, gesture=None, palm=False, flicker_every=0, fist=False):
        end = self.t + ms
        i = 0
        while self.t < end:
            self.t += 33
            i += 1
            g = None if (flicker_every and i % flicker_every == 0) else gesture
            p = palm and not (flicker_every and i % flicker_every == 0)
            f = self.tr.update(g, p, self.t, fist=fist)
            if f:
                self.fired.append((f["name"], round(self.t)))
        return self


cfg = copy.deepcopy(load_config())
cfg["trigger"]["armTimeoutMs"] = 5000  # tests below assume a 5 s timeout

s = Sim(cfg).run(2000, UP)
check("gesture without arming does nothing", s.fired == [] and s.tr.state == DISARMED)

s = Sim(cfg).run(600, palm=True)
check("palm 0.6 s is not enough to arm", s.tr.state == DISARMED)

s = Sim(cfg).run(1100, palm=True)
check("palm 1 s arms", s.tr.state == ARMED)
s.run(300, UP)
check("gesture held 0.3 s does not fire", s.fired == [])
s.run(300, UP)
check("gesture held 0.5 s fires once", [f[0] for f in s.fired] == ["volume up"], str(s.fired))
s.run(3000, UP)
check("keep holding: no repeat (repeatWhileHeld false)", len(s.fired) == 1)
s.run(200, None).run(600, UP)
check("release + hold again fires again", len(s.fired) == 2, str(s.fired))
s.run(600, YT)
check("switch to other gesture fires it", s.fired[-1][0] == "open youtube")

s = Sim(cfg).run(1100, palm=True).run(600, UP, flicker_every=4)
check("1-frame tracking flicker doesn't reset hold", len(s.fired) == 1, str(s.fired))

s = Sim(cfg).run(1100, palm=True).run(5200, None)
check("armed times out after 5 s of nothing", s.tr.state == DISARMED)
s.run(600, UP)
check("after timeout, gesture does nothing", s.fired == [])

s = Sim(cfg).run(1100, palm=True).run(4000, None).run(500, palm=True).run(3000, None)
check("palm again extends armed time", s.tr.state == ARMED)

s = Sim(cfg).run(1100, palm=True)
s.run(600, UP)
t_fire = s.fired[0][1]
s.run(100, None).run(300, YT)   # within cooldown window
check("cooldown blocks immediate second action", len(s.fired) == 1, str(s.fired))
s.run(400, YT)
check("after cooldown the next action works", len(s.fired) == 2, str(s.fired))

rcfg = copy.deepcopy(cfg)
rcfg["trigger"]["repeatWhileHeld"] = True
s = Sim(rcfg).run(1100, palm=True).run(3000, UP)
check("repeatWhileHeld: holding repeats", len(s.fired) >= 3, f"{len(s.fired)} fires in 3 s")

# --- fist disarm
s = Sim(cfg).run(1100, palm=True).run(300, fist=True)
check("fist 0.3 s does not disarm yet", s.tr.state == ARMED)
s.run(300, fist=True)
check("fist 0.5 s disarms", s.tr.state == DISARMED)
s.run(1000, UP)
check("after fist-disarm, gestures do nothing", s.fired == [])
s.run(1100, palm=True).run(600, UP)
check("palm re-arms after fist", [f[0] for f in s.fired] == ["volume up"])

s = Sim(cfg).run(1100, palm=True).run(2000, fist=True)
check("fist never fires as a gesture", s.fired == [])

s = Sim(cfg).run(1100, palm=True).run(600, UP)
s.run(100, fist=True)  # fist during cooldown
s.run(500, fist=True)
check("fist disarms during cooldown too", s.tr.state == DISARMED)

s = Sim(cfg).run(2000, fist=True)
check("fist while disarmed: stays disarmed", s.tr.state == DISARMED)

print("ALL PASS" if not fails else f"{fails} FAILED")
