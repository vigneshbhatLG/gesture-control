"""Analyse logs/pointer_gaps*.csv to calibrate the thumb-tap click.

Prints the thumb-gap distribution, how often the gap crossed the thresholds,
and clicks per minute. Run: .venv\\Scripts\\python tools\\gap_stats.py [file.csv]
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "pointer_gaps.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))
if not rows:
    sys.exit(f"no rows in {path}")

t = np.array([float(r["t_ms"]) for r in rows])
raw = np.array([float(r["gap_raw"]) for r in rows])
med = np.array([float(r["gap_median"]) for r in rows])
clicks = int(sum(int(r["click"]) for r in rows))
dt = np.diff(t)
active_s = float(dt[dt < 200].sum()) / 1000  # ignore gaps between sessions

print(f"{path.name}: {len(rows)} frames, ~{active_s:.0f} s in pointer mode, "
      f"{clicks} clicks ({clicks / max(active_s, 1) * 60:.0f}/min)")
for name, a in (("raw", raw), ("median", med)):
    p = np.percentile(a, [5, 25, 50, 75, 95])
    print(f"  gap {name:6s} p5 {p[0]:.2f}  p25 {p[1]:.2f}  median {p[2]:.2f}  "
          f"p75 {p[3]:.2f}  p95 {p[4]:.2f}")
jitter = np.abs(np.diff(raw))
print(f"  frame-to-frame raw jitter: median {np.median(jitter):.3f}, p95 "
      f"{np.percentile(jitter, 95):.3f}")
hist, edges = np.histogram(raw, bins=np.arange(0, 1.05, 0.05))
print("  raw gap histogram:")
for h, e in zip(hist, edges):
    print(f"    {e:.2f}-{e + 0.05:.2f} {'#' * int(60 * h / max(hist.max(), 1))} {h}")
