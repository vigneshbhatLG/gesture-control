"""Tests: binder (bind / move / unbind), per-action repeat, action dispatch (dry run)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import actions
from core.binder import Binder
from core.config import load_config
from core.trigger import Trigger

fails = 0


def check(name, cond, extra=""):
    global fails
    fails += not cond
    print("PASS" if cond else "FAIL", name, extra)


UP = {"id": "g_up", "name": "thumb"}
YT = {"id": "g_yt", "name": "rock"}

# --- binder
gmap = {}
b = Binder()
b.start(UP)
c, ch = b.key(ord("1"), 0, gmap)
check("bind 1 -> volumeUp", gmap == {"g_up": "volumeUp"} and c and ch, str(gmap))
b.start(YT)
b.key(ord("1"), 0, gmap)
check("same action on other gesture moves it", gmap == {"g_yt": "volumeUp"}, b.message)
b.start(YT)
b.key(ord("7"), 0, gmap)
check("rebind gesture to openYouTube", gmap == {"g_yt": "openYouTube"})
b.start(YT)
b.key(ord("0"), 0, gmap)
check("0 unbinds", gmap == {})
b.start(UP)
c, ch = b.key(ord("q"), 0, gmap)
check("other keys swallowed while menu open (q doesn't quit)", c and not ch and b.active)
b.key(27, 0, gmap)
check("Esc closes menu", not b.active)
check("menu lists all actions", len(b.lines(0, gmap)) >= 1 and
      all(actions.label(a) in " ".join((b.start(UP), b.lines(0, gmap))[1]) for a in actions.ORDER))
b.key(27, 0, gmap)

# --- per-action repeat through the trigger
gmap = {"g_up": "volumeUp", "g_yt": "openYouTube"}
cfg = load_config()
tr = Trigger(cfg, repeat_for=lambda g: actions.repeats(gmap.get(g["id"])))


def run(ms, g=None, palm=False, t=[0.0]):
    fired = []
    end = t[0] + ms
    while t[0] < end:
        t[0] += 33
        f = tr.update(g, palm, t[0])
        if f:
            fired.append(f["name"])
    return fired


run(1100, palm=True)
f = run(2000, UP)
check("volume (repeat) fires repeatedly while held", len(f) >= 5, f"{len(f)} in 2 s")
run(200)
f = run(2000, YT)
check("open YouTube (no repeat) fires once while held", f == ["rock"], str(f))

# --- dispatch
check("known action dispatches (dry run)", actions.run("volumeUp", dry_run=True))
check("unknown action ignored", not actions.run("formatHardDrive", dry_run=True))
check("unbound gesture -> no action", not actions.run(None, dry_run=True))

print("ALL PASS" if not fails else f"{fails} FAILED")
