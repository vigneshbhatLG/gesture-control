"""Step 2: gesture zone + hand segmentation.

Keys:  q quit | b relearn background | m cycle mode (diff/skin/both) | s save settings
Slider on the "mask" window: difference threshold.
"""
import time
import cv2
import numpy as np

from core.config import load_config, save_config
from core.camera import Camera
from core.segment import HandSegmenter
from core.face import FaceMasker, rects_to_roi


def roi_rect(frame, roi):
    h, w = frame.shape[:2]
    return (int(roi["x"] * w), int(roi["y"] * h),
            int(roi["w"] * w), int(roi["h"] * h))


def put(frame, text, y, color=(0, 255, 255), scale=0.5):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def debug_view(seg, mask):
    """Side by side: what 'diff' sees | what 'skin' sees | final mask, 2x size."""
    if seg.debug is None:
        panels = [mask, mask, mask]
    else:
        panels = [seg.debug[0], seg.debug[1], mask]
    view = cv2.cvtColor(np.hstack(panels), cv2.COLOR_GRAY2BGR)
    view = cv2.resize(view, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    pw = view.shape[1] // 3
    for i, label in enumerate(("diff", "skin", f"final ({seg.mode})")):
        cv2.putText(view, label, (i * pw + 5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
        if i:
            cv2.line(view, (i * pw, 0), (i * pw, view.shape[0]), (0, 0, 255), 1)
    return view


def set_thresh(seg, v):
    seg.thresh = max(v, 1)


def main():
    cfg = load_config()
    cam = Camera(cfg)
    seg = HandSegmenter(cfg)
    facer = FaceMasker(cfg)

    # Fixed positions so the windows don't overlap / open off-screen
    cv2.namedWindow("gesture")
    cv2.moveWindow("gesture", 40, 40)
    cv2.namedWindow("mask")
    cv2.moveWindow("mask", 40, 340)
    cv2.createTrackbar("threshold", "mask", seg.thresh, 100,
                       lambda v: set_thresh(seg, v))
    prev = time.time()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Frame grab failed")
                break

            x, y, w, h = roi_rect(frame, cfg["roi"])
            faces = facer.update(frame)
            mask, hand = seg.process(frame[y:y + h, x:x + w],
                                     exclude=rects_to_roi(faces, x, y))

            # Excluded face regions (blue)
            for fx, fy, fw, fh in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 1)

            # Gesture zone: green = hand found, orange = no hand
            zone_color = (0, 255, 0) if hand is not None else (0, 140, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), zone_color, 2)

            if seg.calibrating:
                put(frame, "Learning background - keep hand OUT of box",
                    frame.shape[0] - 10, (0, 0, 255), 0.4)
            elif hand is not None:
                # Contour + box that follows the hand (offset back to full frame)
                cv2.drawContours(frame, [hand], -1, (255, 0, 255), 2,
                                 offset=(x, y))
                hx, hy, hw, hh = cv2.boundingRect(hand)
                cv2.rectangle(frame, (x + hx, y + hy),
                              (x + hx + hw, y + hy + hh), (255, 255, 0), 1)
                m = cv2.moments(hand)
                if m["m00"]:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    cv2.circle(frame, (x + cx, y + cy), 4, (0, 0, 255), -1)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            put(frame, f"FPS {fps:.0f}  mode {seg.mode}  thr {seg.thresh}", 20)
            put(frame, seg.status, 40,
                (0, 255, 0) if hand is not None else (0, 140, 255))

            cv2.imshow("gesture", frame)
            cv2.imshow("mask", debug_view(seg, mask))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                seg.reset_background()
            elif key == ord("m"):
                seg.next_mode()
            elif key == ord("s"):
                cfg["segment"]["diffThreshold"] = seg.thresh
                cfg["segment"]["mode"] = seg.mode
                save_config(cfg)
                print("Saved settings to config.json")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
