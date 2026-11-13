"""Print spread (consistency) and separation of recorded gestures.

Use it to calibrate thresholds in config.json -> gestures:
  matchThreshold / consistencyMax  should sit above the spreads,
  distinctMin                      should sit below the separations.
Run: .venv\\Scripts\\python tools\\gesture_stats.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.config import load_config  # noqa: E402
from core.gestures import (GestureStore, distance, is_fist,  # noqa: E402
                           is_open_palm, is_pointing, thumb_gap)

cfg = load_config()
store = GestureStore(cfg, ROOT)
g = cfg["gestures"]
print(f"thresholds: match <= {g['matchThreshold']}, consistency <= "
      f"{g['consistencyMax']}, distinct >= {g['distinctMin']}\n")

print("Spread within each gesture (same gesture, different samples):")
for ges in store.gestures:
    d = [distance(a, b) for a, b in combinations(ges["samples"], 2)]
    print(f"  {ges['name']:14s} mean {np.mean(d):.3f}  max {np.max(d):.3f}")

print("\nSeparation between gestures (closest pair of samples):")
for a, b in combinations(store.gestures, 2):
    d = [distance(x, y) for x in a["samples"] for y in b["samples"]]
    print(f"  {a['name']} vs {b['name']}: min {min(d):.3f}  mean {np.mean(d):.3f}")

print("\nConflicts with built-in poses (should all be 0/5):")
for ges in store.gestures:
    s = ges["samples"]
    gaps = ", ".join(f"{thumb_gap(x):.2f}" for x in s)
    print(f"  {ges['name']:14s} open palm {sum(map(is_open_palm, s))}/{len(s)}  "
          f"fist {sum(map(is_fist, s))}/{len(s)}  pointing {sum(map(is_pointing, s))}/{len(s)}"
          f"  thumb gap [{gaps}] (fist needs < 0.4)")

size = store.path.stat().st_size if store.path.exists() else 0
n = sum(len(x["samples"]) for x in store.gestures)
print(f"\n{store.path.name}: {size} bytes, {len(store.gestures)} gestures, {n} samples "
      f"(TV binary float16 would be {n * 84} bytes)")
