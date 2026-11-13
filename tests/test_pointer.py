"""Tests: One Euro filter, pointer activation/exit, relative motion, thumb-tap click."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from core.config import load_config
from core.pointer import OneEuro, Pointer

fails = 0


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print("PASS" if cond else "FAIL", name, extra)


# --- One Euro: smooths jitter at rest, follows fast motion
rng = np.random.default_rng(0)
f = OneEuro(1.0, 0.01)
raw = [500 + rng.normal(0, 5) for _ in range(90)]
out = [f(x, i / 30) for i, x in enumerate(raw)]
check("filter reduces jitter at rest", np.std(out[30:]) < np.std(raw[30:]) / 2,
      f"raw std {np.std(raw[30:]):.1f} -> {np.std(out[30:]):.1f}")
f = OneEuro(1.0, 0.01)
for i in range(30):
    f(0, i / 30)
for i in range(30, 45):
    y = f((i - 29) * 40, i / 30)  # fast move: 1200 px/s
check("filter follows fast motion (small lag)", abs(y - 600) < 120, f"target 600 got {y:.0f}")


# --- Pointer state machine with fake hands
def fake(knuckle=(0.5, 0.5), gap=0.8, palm=0.1):
    """raw: 21 points with landmark 5 at `knuckle`, palm length `palm` (wrist 0 ->
    middle knuckle 9, image units); norm: thumb gap controlled."""
    raw = [(0.5, 0.6)] * 21
    raw[0] = (0.5, 0.6 + palm)
    raw[9] = (0.5, 0.6)
    raw[5] = knuckle
    norm = np.zeros((21, 2), np.float32)
    for j in (5, 6, 9, 10, 13):
        norm[j] = (0.0, -1.0)
    norm[4] = (gap, -1.0)  # thumb tip `gap` palm lengths from those points
    return {"raw": raw, "norm": norm}


cfg = load_config()
cursor_start = (1000, 500)
p = Pointer(cfg, 1920, 1080, 320 / 240, get_cursor=lambda: cursor_start)
t = 0.0


def step(n, hand, pointing=True, fist=False):
    global t
    evs = []
    for _ in range(n):
        t += 33
        evs.append(p.update(hand, t, pointing=pointing, fist=fist))
    return evs


step(5, fake(), pointing=False)
step(5, fake(), pointing=True)  # ~165 ms
check("pointing < 300 ms doesn't activate", not p.active)
step(6, fake(), pointing=True)
check("pointing 300 ms activates", p.active)
check("starts at current OS cursor", p.cursor == cursor_start, str(p.cursor))

# move knuckle right by 0.1 frame widths over 10 frames
for i in range(1, 11):
    step(1, fake(knuckle=(0.5 + 0.01 * i, 0.5)))
step(20, fake(knuckle=(0.6, 0.5)))  # let filter settle
moved = p.cursor[0] - cursor_start[0]
expected = 0.1 * (320 / 240) * p.gain * 1080
check("cursor moves right, ~expected distance", abs(moved - expected) < expected * 0.15,
      f"moved {moved:.0f}px, expected ~{expected:.0f}px")
check("vertical stays put", abs(p.cursor[1] - cursor_start[1]) < 5)

# click: initial state pressed=True -> must open first (gap 0.8 > 0.55 already did)
evs = step(6, fake(knuckle=(0.6, 0.5), gap=0.2))
check("thumb tap -> exactly one click", sum(e["click"] for e in evs) == 1)
before = p.cursor
step(5, fake(knuckle=(0.65, 0.5), gap=0.2))
check("cursor frozen while thumb pressed", p.cursor == before)
evs = step(10, fake(knuckle=(0.65, 0.5), gap=0.2))
check("holding thumb down: no repeat clicks", sum(e["click"] for e in evs) == 0)
evs = step(3, fake(knuckle=(0.65, 0.5), gap=0.45))  # between thresholds: still pressed
evs += step(3, fake(knuckle=(0.65, 0.5), gap=0.3))
check("hysteresis: half-open thumb doesn't re-click", sum(e["click"] for e in evs) == 0)
step(8, fake(knuckle=(0.65, 0.5), gap=0.8))
evs = step(6, fake(knuckle=(0.65, 0.5), gap=0.2))
check("open again + tap -> second click", sum(e["click"] for e in evs) == 1)

# jitter: open thumb with random 1-frame dips below pressGap must NOT click
step(8, fake(knuckle=(0.65, 0.5), gap=0.8))
evs = []
for i in range(60):
    evs += step(1, fake(knuckle=(0.65, 0.5), gap=0.2 if i % 4 == 0 else 0.7))
check("1-frame thumb jitter doesn't click (was ~85 false clicks)", sum(e["click"] for e in evs) == 0,
      f"{sum(e['click'] for e in evs)} clicks")

# clamp at screen edge
step(5, fake(gap=0.8))
for i in range(30):
    step(1, fake(knuckle=(0.65 + 0.05 * i, 0.5), gap=0.8))
check("cursor clamped to screen", 0 <= p.cursor[0] <= 1919, str(p.cursor))

# open palm (not pointing): cursor frozen, exits after 300 ms
step(11, fake(), pointing=True)
p.stop(); step(11, fake(), pointing=True)
before = p.cursor
step(5, fake(knuckle=(0.2, 0.2)), pointing=False)       # ~165 ms, hand moves a lot
check("open palm: cursor does NOT move", p.cursor == before, f"{before} -> {p.cursor}")
check("open palm < 300 ms: still active", p.active)
step(3, fake(knuckle=(0.2, 0.2)), pointing=True)          # back to pointing
check("resume pointing: no jump", abs(p.cursor[0] - before[0]) < 5 and abs(p.cursor[1] - before[1]) < 5,
      f"{before} -> {p.cursor}")
step(12, fake(), pointing=False)                          # ~400 ms open palm
check("open palm 300 ms exits pointer", not p.active)
step(11, fake(), pointing=True)

# exit: hand lost
step(10, None)
check("brief hand loss keeps pointer", p.active)
step(15, None)
check("hand lost 0.6 s exits", not p.active)

# exit: fist
step(11, fake(), pointing=True)
step(20, fake(), fist=True)
check("fist exits pointer", not p.active)

# --- requireArmed: pointing while disarmed must not start the pointer
p2 = Pointer(cfg, 1920, 1080, 320 / 240, get_cursor=lambda: (100, 100))
tt = 0.0
for _ in range(30):
    tt += 33
    p2.update(fake(), tt, pointing=True, armed=False)
check("requireArmed: pointing while DISARMED does nothing", not p2.active)
for _ in range(11):
    tt += 33
    p2.update(fake(), tt, pointing=True, armed=True)
check("requireArmed: pointing while ARMED starts pointer", p2.active)


# --- adaptive thresholds: the live failure (thumb out only ~0.58, tucked ~0.25)
def taps(pt, out_gap, in_gap, n_taps, t0):
    tt, clicks = t0, 0
    for _ in range(40):  # learn "out" level
        tt += 33
        pt.update(fake(gap=out_gap), tt, pointing=True)
    for _ in range(n_taps):
        for g, frames in ((in_gap, 8), (out_gap, 12)):
            for _ in range(frames):
                tt += 33
                clicks += pt.update(fake(gap=g), tt, pointing=True)["click"]
    return clicks, tt


def fresh(adaptive):
    import copy
    c = copy.deepcopy(cfg)
    c["pointer"]["adaptive"]["enabled"] = adaptive
    c["pointer"]["pressGap"], c["pointer"]["releaseGap"] = 0.30, 0.60
    pt = Pointer(c, 1920, 1080, 320 / 240)
    tt = 0.0
    for _ in range(11):
        tt += 33
        pt.update(fake(gap=0.58), tt, pointing=True)
    return pt, tt


pf, tf = fresh(adaptive=False)
c_fixed, _ = taps(pf, 0.58, 0.25, 5, tf)
pa, ta = fresh(adaptive=True)
c_adapt, ta = taps(pa, 0.58, 0.25, 5, ta)
check("fixed 0.30/0.60 thresholds miss taps when thumb-out is 0.58 (the live bug)",
      c_fixed < 5, f"{c_fixed}/5 clicks")
check("adaptive thresholds: 5 taps -> 5 clicks", c_adapt == 5,
      f"{c_adapt}/5, press {pa.press_gap:.2f} release {pa.release_gap:.2f}")
pb, tb = fresh(adaptive=True)
c_wide, _ = taps(pb, 0.80, 0.25, 5, tb)
check("adaptive also works for the earlier wide thumb (out 0.80)", c_wide == 5,
      f"{c_wide}/5, press {pb.press_gap:.2f} release {pb.release_gap:.2f}")
# hold the click (thumb tucked) ~5 s: thresholds must not drift, exactly 1 click
before = pa.release_gap
extra = 0
for i in range(158):
    ta += 33
    g = 0.25 if i < 8 else 0.25 + 0.04 * ((i % 3) - 1)
    extra += pa.update(fake(gap=g), ta, pointing=True)["click"]
check("holding click 5 s: thresholds frozen, only 1 click",
      extra == 1 and abs(pa.release_gap - before) < 0.02,
      f"clicks {extra}, release {before:.2f}->{pa.release_gap:.2f}")

# --- scroll (✌️, joystick)
from core.pointer import MOVE, SCROLL


def scroll_run(ps, frames, knuckle, palm=0.1, two=True, pointing=False):
    global ts
    wx = wy = 0
    for _ in range(frames):
        ts += 33
        e = ps.update(fake(knuckle=knuckle, palm=palm), ts, pointing=pointing,
                      two_fingers=two)
        wx += e["scroll"][0]
        wy += e["scroll"][1]
    return wx, wy


ps = Pointer(cfg, 1920, 1080, 320 / 240, get_cursor=lambda: (800, 400))
ts = 0.0
for _ in range(30):
    ts += 33
    ps.update(fake(), ts, two_fingers=True, armed=False)
check("scroll: V while DISARMED does nothing", not ps.active)
scroll_run(ps, 11, (0.5, 0.5))
check("scroll: V while armed starts pointer in SCROLL mode", ps.active and ps.mode == SCROLL)
c0 = ps.cursor
wx, wy = scroll_run(ps, 30, (0.505, 0.505))  # 0.05 palm: inside deadzone
check("scroll: hand near centre (deadzone) -> no scroll", (wx, wy) == (0, 0), str((wx, wy)))
wx, wy = scroll_run(ps, 30, (0.5, 0.45))     # hand UP 0.5 palm for ~1 s
check("scroll: hand up -> scrolls up (wheel +)", wy > 0 and wx == 0, f"wy={wy}")
up_half = wy
wx, wy = scroll_run(ps, 30, (0.5, 0.42))     # 0.8 palm up = full speed
check("scroll: further from centre -> faster", wy > up_half * 1.3, f"{up_half} -> {wy}")
check("scroll: full speed ~ maxNotchesPerSec", abs(wy / 120 - 12) < 3, f"{wy / 120:.1f} notches/s")
wx, wy = scroll_run(ps, 30, (0.5, 0.55))
check("scroll: hand down -> scrolls down (wheel -)", wy < 0 and wx == 0, f"wy={wy}")
scroll_run(ps, 10, (0.5 + 0.05 * 240 / 320, 0.5))  # let smoothing settle first
wx, wy = scroll_run(ps, 30, (0.5 + 0.05 * 240 / 320, 0.5))  # 0.5 palm right (aspect)
check("scroll: hand right -> scrolls right", wx > 0 and wy == 0, f"wx={wx} wy={wy}")
scroll_run(ps, 10, (0.5 - 0.05 * 240 / 320, 0.5))
wx, wy = scroll_run(ps, 30, (0.5 - 0.05 * 240 / 320, 0.5))
check("scroll: hand left -> scrolls left", wx < 0 and wy == 0, f"wx={wx} wy={wy}")
wx, wy = scroll_run(ps, 30, (0.5 + 0.03 * 240 / 320, 0.44))  # mostly up, a bit right
check("scroll: axis lock (diagonal -> only the main direction)", wy > 0 and wx == 0,
      f"wx={wx} wy={wy}")
check("scroll: cursor frozen while scrolling", ps.cursor == c0)

# distance independence: hand twice as big (closer), same offset in palm lengths
pn = Pointer(cfg, 1920, 1080, 320 / 240)
ts = 0.0
scroll_run(pn, 11, (0.5, 0.5), palm=0.2)
wx, wy = scroll_run(pn, 30, (0.5, 0.40), palm=0.2)  # 0.1 raw = 0.5 palm at palm 0.2
check("scroll: same speed when closer to the camera (palm-length units)",
      abs(wy - up_half) < up_half * 0.15, f"{up_half} vs {wy}")

# back to ☝️: cursor mode, no jump, no accidental click
ts_before = ps.cursor
clicks = 0
for _ in range(15):
    ts += 33
    e = ps.update(fake(knuckle=(0.6, 0.3), gap=0.2), ts, pointing=True)
    clicks += e["click"]
check("scroll -> 1 finger: back to MOVE mode", ps.mode == MOVE)
check("scroll -> 1 finger: cursor doesn't jump", ps.cursor == ts_before, f"{ts_before} -> {ps.cursor}")
check("scroll -> 1 finger: no accidental click (thumb must open first)", clicks == 0)
scroll_run(ps, 15, (0.6, 0.3), two=False)  # open palm etc.
check("scroll: neither 1 nor 2 fingers 0.3 s -> exits", not ps.active)

# disabled
p.enabled = False
step(20, fake(), pointing=True)
check("disabled -> never activates", not p.active)

print("ALL PASS" if not fails else f"{fails} FAILED")
