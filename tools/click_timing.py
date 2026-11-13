"""Show each click with the thumb gap around it, to tell real taps from jitter.

Real tap: gap clearly drops (e.g. 0.5 -> 0.2) for a few frames, then rises.
Jitter:   tiny dip just under the press mark, or clicks every ~300 ms in a burst.
Run: .venv\\Scripts\\python tools\\click_timing.py [file.csv]
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "pointer_gaps.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))
idx = [i for i, r in enumerate(rows) if r["click"] == "1"]
prev_t = None
print(f"{len(idx)} clicks in {path.name}")
print("  #   t(s)   since prev   gap before -> min during -> after   tap depth")
for n, i in enumerate(idx, 1):
    t = float(rows[i]["t_ms"]) / 1000
    before = [float(r["gap_median"]) for r in rows[max(0, i - 12):i - 3]]
    during = [float(r["gap_median"]) for r in rows[i - 3:i + 6]]
    after = [float(r["gap_median"]) for r in rows[i + 6:i + 18]]
    b = max(before) if before else float("nan")
    d = min(during)
    a = max(after) if after else float("nan")
    since = f"{t - prev_t:6.2f}s" if prev_t is not None else "      -"
    print(f"  {n:<3} {t:6.1f}  {since}     {b:.2f} -> {d:.2f} -> {a:.2f}        {b - d:.2f}")
    prev_t = t
