"""Run every test script. Usage: .venv\\Scripts\\python tests\\run_all.py"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
total_fail = 0
for t in sorted(HERE.glob("test_*.py")):
    res = subprocess.run([sys.executable, str(t)], capture_output=True, text=True)
    out = res.stdout
    if res.returncode != 0:  # a crash must count as a failure, not "0 pass"
        out += "\nFAIL script crashed: " + (res.stderr.strip().splitlines() or ["?"])[-1]
    p = sum(1 for line in out.splitlines() if line.startswith("PASS"))
    f = [line for line in out.splitlines() if line.startswith("FAIL")]
    k = [line for line in out.splitlines() if line.startswith("KNOWN")]
    total_fail += len(f)
    extra = f"  ({len(k)} known limitation)" if k else ""
    print(f"{t.name:22s} {p:3d} pass  {len(f)} fail{extra}")
    for line in f + k:
        print("    " + line)
sys.exit(1 if total_fail else 0)
