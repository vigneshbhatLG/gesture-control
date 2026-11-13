"""Settings loader/saver.

Key names here intentionally match the future webOS settingsservice keys
(category "gesture"), so the TV port only swaps *where* values come from.
"""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
