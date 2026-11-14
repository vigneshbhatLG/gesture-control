"""What a gesture can DO. Gestures are bound to these via config `gestureMap`
(gesture id -> action id) - the same thing the Enact Settings screen will edit.

Laptop prototype: Windows media keys (ctypes, no extra deps) + browser.
TV (later):       each action becomes a platform call - see the "tv" field.
                  (Exact Luna APIs to be confirmed with the platform team.)

repeat=True  -> keeps firing while the gesture is held (volume)
repeat=False -> fires once per hold (open an app)
"""
import sys
import webbrowser

KEYEVENTF_KEYUP = 0x0002


def _press(vk):
    if sys.platform != "win32":
        print(f"(media key 0x{vk:02X} - only implemented on Windows)")
        return
    import ctypes
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


ACTIONS = {
    "volumeUp":    {"label": "Volume up",      "repeat": True,
                    "run": lambda: _press(0xAF), "tv": "remote key VOLUMEUP"},
    "volumeDown":  {"label": "Volume down",    "repeat": True,
                    "run": lambda: _press(0xAE), "tv": "remote key VOLUMEDOWN"},
    "mute":        {"label": "Mute",           "repeat": False,
                    "run": lambda: _press(0xAD), "tv": "remote key MUTE"},
    "playPause":   {"label": "Play / pause",   "repeat": False,
                    "run": lambda: _press(0xB3), "tv": "remote key PLAY/PAUSE"},
    "nextTrack":   {"label": "Next",           "repeat": False,
                    "run": lambda: _press(0xB0), "tv": "remote key FASTFORWARD / CH+"},
    "prevTrack":   {"label": "Previous",       "repeat": False,
                    "run": lambda: _press(0xB1), "tv": "remote key REWIND / CH-"},
    "openYouTube": {"label": "Open YouTube",   "repeat": False,
                    "run": lambda: webbrowser.open("https://www.youtube.com"),
                    "tv": "applicationManager launch (YouTube app id)"},
    # Special: handled by the tracker (ends pointer mode), not run as a command.
    # Only matched while pointer mode is active.
    "exitPointer": {"label": "Exit pointer",   "repeat": False,
                    "run": lambda: None, "tv": "(internal) hide pointer"},
    # No Windows media key for seeking. YouTube: J / L = exactly 10 s back /
    # forward (arrow keys are only 5 s there). Video window must be focused.
    "seekBack10":    {"label": "Back 10 s",    "repeat": False,
                      "run": lambda: _press(0x4A),  # J
                      "tv": "media seek -10 s (remote REWIND in most apps)"},
    "seekForward10": {"label": "Skip 10 s",    "repeat": False,
                      "run": lambda: _press(0x4C),  # L
                      "tv": "media seek +10 s (remote FASTFORWARD in most apps)"},
}
ORDER = list(ACTIONS)  # numbering used by the bind menu (1..N)


def label(action_id):
    return ACTIONS[action_id]["label"] if action_id in ACTIONS else "-"


def repeats(action_id):
    return action_id in ACTIONS and ACTIONS[action_id]["repeat"]


def run(action_id, dry_run=False):
    """Execute an action. Unknown ids are ignored (settings validation rule)."""
    if action_id not in ACTIONS:
        return False
    if not dry_run:
        ACTIONS[action_id]["run"]()
    return True
