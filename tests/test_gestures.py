"""Synthetic tests for normalisation, matching, record checks, persistence, recorder."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import copy

import numpy as np

from core.config import load_config
from core.gestures import GestureStore, distance, normalize
from core.recorder import Recorder, OFF

import tempfile
SCRATCH = Path(tempfile.gettempdir())
rng = np.random.default_rng(1)
ASPECT = 320 / 240
fails = 0


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print("PASS" if cond else "FAIL", name, extra)


def hand(fingers_up=(1, 1, 1, 1, 1), cx=0.5, cy=0.6, size=0.12, noise=0.0, rot=0.0):
    """Fake right hand in image coords (x right, y down). fingers: thumb..pinky."""
    pts = np.zeros((21, 2))
    base_x = [-0.6, -0.35, 0.0, 0.3, 0.55]   # MCP x positions
    for f in range(5):
        mcp = 1 + f * 4
        x0 = base_x[f]
        pts[mcp] = (x0, -0.9 if f else -0.4)
        for j in range(1, 4):
            if fingers_up[f]:
                pts[mcp + j] = (x0 * (1 + 0.1 * j), pts[mcp][1] - 0.35 * j)
            else:  # curled: tips fold back down toward palm
                pts[mcp + j] = (x0 * 0.9, pts[mcp][1] + 0.15 * j)
    pts[9] = (0.0, -1.0)  # middle MCP defines palm length = 1
    c, s = np.cos(rot), np.sin(rot)
    pts = pts @ np.array([[c, -s], [s, c]]).T
    pts += rng.normal(0, noise, pts.shape)
    img = pts * size
    img[:, 0] /= ASPECT
    img += (cx, cy)
    return [tuple(p) for p in img]


n = lambda pts, label="Right": normalize(pts, ASPECT, label, True)

# --- normalisation
a = n(hand())
check("shift + scale invariant", distance(a, n(hand(cx=0.3, cy=0.4, size=0.2))) < 1e-4)
mirrored = [(1 - x, y) for x, y in hand()]  # same hand seen as left
check("left hand mirrored == right", distance(a, n(mirrored, "Left")) < 1e-4)
check("rotation NOT ignored (thumbs up vs down)",
      distance(a, n(hand(rot=np.pi))) > 1.0)

# --- distances: noise vs different pose
same = np.mean([distance(a, n(hand(noise=0.04))) for _ in range(20)])
rock = (0, 1, 0, 0, 1)   # 🤘 (✌️ is reserved for scroll now)
rock3 = (0, 1, 0, 1, 1)  # 🤘 + ring finger: one finger different
diff_one = distance(n(hand(rock)), n(hand(rock3)))
print(f"     same pose w/ noise: {same:.3f}   rock vs rock+ring (1 finger): {diff_one:.3f}")
check("same pose under matchThreshold 0.25", same < 0.25)
check("one-finger difference over distinctMin 0.3", diff_one > 0.3)

# --- store: record checks + match + persistence
cfg = copy.deepcopy(load_config())
cfg["gestures"]["file"] = str(SCRATCH / "gestures_test.json")
(SCRATCH / "gestures_test.json").unlink(missing_ok=True)
store = GestureStore(cfg, SCRATCH)

samples = [n(hand(rock, noise=0.03)) for _ in range(5)]
ok, msg = store.check_new(samples)
check("consistent samples accepted", ok, msg)
store.add("rock", samples)

sloppy = [n(hand(f, noise=0.03)) for f in [rock, rock3, (1, 1, 1, 1, 1), (0, 0, 0, 0, 0), rock]]
ok, msg = store.check_new(sloppy)
check("inconsistent samples rejected", not ok, msg)

ok, msg = store.check_new([n(hand(rock, noise=0.03)) for _ in range(5)])
check("duplicate of existing gesture rejected", not ok, msg)

ok, msg = store.check_new([n(hand((0, 1, 1, 0, 0), noise=0.03)) for _ in range(5)])
check("V (two fingers) recording rejected: reserved for scroll", not ok, msg)

ok, msg = store.check_new([n(hand((0, 0, 0, 0, 0), noise=0.03)) for _ in range(5)])
check("different gesture (fist) accepted", ok, msg)
store.add("fist", [n(hand((0, 0, 0, 0, 0), noise=0.03)) for _ in range(5)])

best, d, _ = store.match(n(hand(rock, noise=0.03, cx=0.2, size=0.08)))
check("live rock (moved, smaller) matches rock", best and best["name"] == "rock", f"d={d:.2f}")
best, d, _ = store.match(n(hand((1, 1, 1, 1, 1))))
check("open palm matches nothing", best is None, f"closest d={d:.2f}")

store2 = GestureStore(cfg, SCRATCH)
check("save/load round trip", [g["name"] for g in store2.gestures] == ["rock", "fist"]
      and distance(store2.gestures[0]["samples"][0], store.gestures[0]["samples"][0]) < 0.01)
size = (SCRATCH / "gestures_test.json").stat().st_size
print(f"     gestures.json: {size} bytes for 2 gestures x 5 samples (JSON text; TV binary ~84 B/sample)")

# --- recorder flow
rec = Recorder(cfg, store)
rec.start()
for ch in "ok":
    rec.key(ord(ch), 0)
rec.key(8, 0)          # backspace -> "o"
rec.key(ord("k"), 0)   # -> "ok"
rec.key(ord("q"), 0)   # 'q' while naming must be a letter, not quit
check("typing incl. backspace and q", rec.name == "okq", rec.name)
rec.key(8, 0)
rec.key(13, 0)         # Enter -> capture starts in 2 s
t = 0
pose = (1, 0, 0, 0, 1)
while rec.active and t < 20000:
    t += 33
    rec.update(n(hand(pose, noise=0.03)) if t % 3000 > 300 else None, t)
check("recorder captured + saved", any(g["name"] == "ok" for g in store.gestures),
      rec.message)
check("recorder back to off", rec.state == OFF)

(SCRATCH / "gestures_test.json").unlink(missing_ok=True)
print("ALL PASS" if not fails else f"{fails} FAILED")
