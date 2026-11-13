"""Laptop output for pointer mode: real Windows mouse via ctypes (no deps).
On the TV this is replaced by webOS pointer injection (API: ask platform team).
"""
import ctypes
import sys

MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_WHEEL, MOUSEEVENTF_HWHEEL = 0x0800, 0x1000


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class Mouse:
    def __init__(self):
        self.ok = sys.platform == "win32"
        if not self.ok:
            return
        self.user32 = ctypes.windll.user32
        # Use real pixels, not DPI-scaled ones, so cursor maths matches the screen
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            self.user32.SetProcessDPIAware()

    def screen_size(self):
        if not self.ok:
            return 1920, 1080
        return self.user32.GetSystemMetrics(0), self.user32.GetSystemMetrics(1)

    def get_pos(self):
        if not self.ok:
            return None
        p = POINT()
        self.user32.GetCursorPos(ctypes.byref(p))
        return p.x, p.y

    def move(self, x, y):
        if self.ok:
            self.user32.SetCursorPos(int(x), int(y))

    def scroll(self, wx, wy):
        """Wheel units (120 = one notch). wy > 0 scrolls up, wx > 0 scrolls right."""
        if not self.ok:
            return
        if wy:
            self.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(wy), 0)
        if wx:
            self.user32.mouse_event(MOUSEEVENTF_HWHEEL, 0, 0, int(wx), 0)

    def click(self):
        if self.ok:
            self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
