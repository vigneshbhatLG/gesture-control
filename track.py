"""MediaPipe hand tracking + custom gestures + arm/hold/cooldown trigger.

Flow: hold open palm 1 s -> ARMED -> hold a recorded gesture 0.5 s -> FIRED
      -> runs the action bound in config.json gestureMap (live-reloaded).

Run with the project venv:  .venv\\Scripts\\python track.py
Keys: q quit | u unload model | r record new gesture | z delete last gesture
      b bind the currently shown gesture to an action (1..N, 0 unbind, Esc)
      p toggle pointer mode feature on/off
Pointer mode: hold ☝️ 0.3 s -> cursor follows index knuckle; thumb tap = click;
              fist or drop hand = exit.

Overlay shows the numbers we need for the TV estimate:
state, FPS, model ms per call, model calls per second, RAM, model load time.
"""
import time
from pathlib import Path

import cv2
import psutil

from core.camera import Camera
from core import actions
from core.binder import Binder
from core.config import CONFIG_PATH, load_config, save_config
from core.gestures import (GestureStore, is_fist, is_open_palm, is_pointing,
                           is_two_fingers, normalize)
from core.mouse import Mouse
from core.pointer import SCROLL, Pointer
from core.hands import CONNECTIONS, HandTracker
from core.recorder import Recorder
from core.trigger import ARMED, COOLDOWN, DISARMED, Trigger
from core.pipeline import IDLE, SEARCH, TRACK, Pipeline

ROOT = Path(__file__).resolve().parent
STATE_COLOR = {IDLE: (160, 160, 160), SEARCH: (0, 200, 255), TRACK: (0, 255, 0)}


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in hand["points"]]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1)
    for i, p in enumerate(pts):
        tip = i in (4, 8, 12, 16, 20)  # fingertips
        cv2.circle(frame, p, 4 if tip else 2, (0, 0, 255) if tip else (255, 0, 255), -1)
    cv2.putText(frame, f'{hand["label"]} {hand["score"]:.2f}',
                (pts[0][0] - 20, pts[0][1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 0), 1)


def put(frame, text, y, color=(0, 255, 255), scale=0.4):
    # Drop shadow (same thickness, offset 1px) so text is readable on bright
    # backgrounds. A thick outline drifts: OpenCV's glyph spacing depends on
    # thickness, so the two layers don't line up.
    cv2.putText(frame, text, (9, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA)


TRIGGER_COLOR = {DISARMED: (160, 160, 160), ARMED: (0, 255, 0),
                 COOLDOWN: (0, 200, 255)}


def draw_trigger(view, trigger, now, gmap):
    """Top-right panel: arm state, hold progress bar, last fired gesture."""
    w = view.shape[1]
    x0, y0, bw = w - 310, 16, 290
    state = trigger.state
    if state == ARMED:
        left = max(0.0, (trigger.armed_until - now) / 1000)
        label = f"ARMED ({left:.0f}s) - fist to disarm"
    elif state == DISARMED:
        label = "DISARMED - hold open palm to arm"
    else:
        label = "COOLDOWN"
    cv2.putText(view, label, (x0 + 1, y0 + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(view, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                TRIGGER_COLOR[state], 1, cv2.LINE_AA)
    p = trigger.progress(now)
    if p is not None:
        cv2.rectangle(view, (x0, y0 + 8), (x0 + bw, y0 + 18), (60, 60, 60), -1)
        if trigger.cand == "fist":
            bar = (0, 0, 255)          # red: disarming
        elif state == DISARMED:
            bar = (255, 255, 0)        # cyan: arming
        else:
            bar = TRIGGER_COLOR[state]  # green: firing a gesture
        cv2.rectangle(view, (x0, y0 + 8), (x0 + int(bw * p), y0 + 18), bar, -1)
    if trigger.last_fired and now - trigger.last_fired[1] < 1500:
        g = trigger.last_fired[0]
        txt = f"FIRED: {actions.label(gmap.get(g['id'])) if g['id'] in gmap else g['name'] + ' (unbound)'}"
        cv2.putText(view, txt, (x0 + 1, y0 + 41), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(view, txt, (x0, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)


def draw_scroll(view, pointer, x0, y0):
    """Joystick: grey ring = deadzone, outer ring = full speed, dot = hand."""
    label = "SCROLL - move hand to scroll, 1 finger = cursor"
    for off, col in ((1, (0, 0, 0)), (0, (0, 255, 255))):
        cv2.putText(view, label, (x0 + off, y0 + off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)
    r_full = 45
    cx, cy = x0 + 60, y0 + 70
    r_dead = int(r_full * pointer.deadzone / pointer.max_offset)
    cv2.circle(view, (cx, cy), r_full, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(view, (cx, cy), r_dead, (120, 120, 120), -1, cv2.LINE_AA)
    ox, oy = pointer.scroll_offset
    px = int(cx + max(-1.2, min(1.2, ox / pointer.max_offset)) * r_full)
    py = int(cy + max(-1.2, min(1.2, oy / pointer.max_offset)) * r_full)
    cv2.line(view, (cx, cy), (px, py), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(view, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)
    vx, vy = pointer.scroll_vel
    arrows = []
    if vy > 0:
        arrows.append(f"UP {vy:.0f}/s")
    elif vy < 0:
        arrows.append(f"DOWN {-vy:.0f}/s")
    if vx > 0:
        arrows.append(f"RIGHT {vx:.0f}/s")
    elif vx < 0:
        arrows.append(f"LEFT {-vx:.0f}/s")
    txt = "  ".join(arrows) or "stopped (centre)"
    cv2.putText(view, txt, (cx + r_full + 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_pointer(view, pointer, now, last_click_ui):
    """Top-right: pointer state + thumb-gap meter (for calibrating the click)."""
    w = view.shape[1]
    x0, y0, bw = w - 310, 16, 290
    if pointer.mode == SCROLL:
        draw_scroll(view, pointer, x0, y0)
        return
    label = "POINTER - thumb tap = click, V = scroll, fist = exit"
    for off, col in ((1, (0, 0, 0)), (0, (255, 0, 255))):
        cv2.putText(view, label, (x0 + off, y0 + off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)
    # thumb gap meter: 0 .. 1 palm lengths, marks at press / release thresholds
    gy = y0 + 10
    cv2.rectangle(view, (x0, gy), (x0 + bw, gy + 10), (60, 60, 60), -1)
    if pointer.gap is not None:
        g = min(pointer.gap, 1.0)
        col = (0, 0, 255) if pointer.pressed else (0, 255, 0)
        cv2.rectangle(view, (x0, gy), (x0 + int(bw * g), gy + 10), col, -1)
    for thr, col in ((pointer.press_gap, (0, 0, 255)), (pointer.release_gap, (0, 255, 0))):
        tx = x0 + int(bw * thr)
        cv2.line(view, (tx, gy - 3), (tx, gy + 13), col, 2)
    gap_txt = f"thumb gap {pointer.gap:.2f}" if pointer.gap is not None else "thumb gap -"
    cv2.putText(view, f"{gap_txt}  ({'pressed' if pointer.pressed else 'open'})",
                (x0, gy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    cx, cy = (int(v) for v in pointer.cursor)
    cv2.putText(view, f"cursor {cx},{cy}", (x0, gy + 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if now - last_click_ui < 600:
        for off, col in ((2, (0, 0, 0)), (0, (0, 255, 255))):
            cv2.putText(view, "CLICK!", (x0 + off, gy + 80 + off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)


def main():
    cfg = load_config()
    cam = Camera(cfg)
    tracker = HandTracker(cfg, ROOT)
    pipe = Pipeline(cfg, tracker)
    store = GestureStore(cfg, ROOT)
    recorder = Recorder(cfg, store)
    binder = Binder()
    gmap = dict(cfg.get("gestureMap", {}))  # gesture id -> action id
    enabled = cfg.get("gestureEnabled", True)
    dry_run = cfg.get("actions", {}).get("dryRun", False)
    # Repeat while held depends on the BOUND ACTION (volume yes, open app no)
    trigger = Trigger(cfg, repeat_for=lambda g: actions.repeats(gmap.get(g["id"])))
    mirror_left = cfg["gestures"]["mirrorLeftHand"]
    aspect = cfg["camera"]["width"] / cfg["camera"]["height"]
    mouse = Mouse()
    sw, sh = mouse.screen_size()
    pointer = Pointer(cfg, sw, sh, aspect, get_cursor=mouse.get_pos)
    move_real = cfg["pointer"].get("moveRealCursor", True)
    real_click = cfg["pointer"].get("realClick", False)
    last_click_ui = -1e9
    # Thumb-gap log for calibrating the click (numbers only, no images)
    gap_log = None
    if cfg["pointer"].get("logGaps"):
        (ROOT / "logs").mkdir(exist_ok=True)
        gap_log = open(ROOT / "logs" / "pointer_gaps.csv", "w", encoding="utf-8",
                       buffering=1)
        gap_log.write("t_ms,gap_raw,gap_median,pressed,click\n")
    proc = psutil.Process()

    cv2.namedWindow("track")
    cv2.moveWindow("track", 40, 40)
    t0 = time.perf_counter()
    prev = t0
    fps = 0.0
    calls_per_s, calls_mark, mark_t = 0.0, 0, t0
    last_key = "-"

    last_match = None
    cfg_mtime = CONFIG_PATH.stat().st_mtime
    cfg_checked = 0.0

    def save_bindings():
        nonlocal cfg_mtime
        cfg["gestureMap"] = gmap
        save_config(cfg)
        cfg_mtime = CONFIG_PATH.stat().st_mtime  # our own write: no reload

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Frame grab failed")
                break
            now = time.perf_counter()
            now_ms = (now - t0) * 1000

            # Live reload: like the TV service's settingsservice subscription,
            # edits to config.json apply without a restart.
            if now - cfg_checked > 1.0:
                cfg_checked = now
                m = CONFIG_PATH.stat().st_mtime
                if m != cfg_mtime:
                    cfg_mtime = m
                    try:
                        cfg = load_config()
                        gmap.clear()
                        gmap.update(cfg.get("gestureMap", {}))
                        trigger.configure(cfg)
                        pointer.configure(cfg)
                        move_real = cfg["pointer"].get("moveRealCursor", True)
                        real_click = cfg["pointer"].get("realClick", False)
                        enabled = cfg.get("gestureEnabled", True)
                        dry_run = cfg.get("actions", {}).get("dryRun", False)
                        binder.say("config.json reloaded", now_ms)
                        print("config.json reloaded", flush=True)
                    except (ValueError, KeyError) as e:  # half-saved / bad JSON
                        binder.say(f"config.json invalid, kept old settings: {e}",
                                   now_ms, 5000)

            if not enabled:
                # Settings toggle OFF: no model, no gestures (TV: service stopped)
                tracker.unload()
                view = cv2.resize(frame, None, fx=2, fy=2)
                put(view, "GESTURE CONTROL OFF  (gestureEnabled = false in config.json)",
                    20, (0, 0, 255), 0.55)
                cv2.imshow("track", view)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            hands = pipe.step(frame, now_ms)

            view = cv2.resize(frame, None, fx=2, fy=2)  # bigger to look at
            for hand in hands:
                draw_hand(view, hand)

            # Normalised first hand -> record or match
            norm = None
            if hands:
                norm = normalize(hands[0]["points"], aspect,
                                 hands[0]["label"], mirror_left)
            if recorder.active:
                recorder.update(norm, now_ms)
            else:
                best, palm, fist, pointing, two = None, False, False, False, False
                if norm is not None:
                    palm = is_open_palm(norm)
                    fist = not palm and is_fist(norm)
                    pointing = not palm and not fist and is_pointing(norm)
                    two = not palm and not fist and not pointing and is_two_fingers(norm)

                # Pointer mode (air mouse) has priority over gestures
                hand_in = ({"raw": hands[0]["points"], "norm": norm}
                           if norm is not None else None)
                # A custom gesture bound to "Exit pointer" ends pointer mode
                # (only checked while the pointer is active)
                exit_match = False
                if pointer.active and norm is not None and "exitPointer" in gmap.values():
                    g_exit, _, _ = store.match(norm)
                    exit_match = bool(g_exit) and gmap.get(g_exit["id"]) == "exitPointer"
                pev = (pointer.update(hand_in, now_ms, pointing=pointing,
                                      fist=fist or exit_match, armed=trigger.armed,
                                      two_fingers=two)
                       if not binder.active else {})
                wx, wy = pev.get("scroll", (0, 0))
                if (wx or wy) and move_real:
                    mouse.scroll(wx, wy)
                if pointer.active:
                    trigger.keep_alive(now_ms)
                if pev.get("reason") == "fist" and fist:
                    trigger.disarm()  # fist = disarm everything, also from pointer
                    print("DISARMED (fist in pointer mode)", flush=True)
                if pointer.active and gap_log and pointer.gap_raw is not None:
                    gap_log.write(f"{now_ms:.0f},{pointer.gap_raw:.3f},{pointer.gap:.3f},"
                                  f"{int(pointer.pressed)},{int(pev.get('click', False))}\n")
                if pointer.active or pev.get("stopped"):
                    if pev.get("moved") and move_real:
                        mouse.move(*pointer.cursor)
                    if pev.get("click"):
                        if real_click:
                            mouse.click()
                        print(f"CLICK at {tuple(int(v) for v in pointer.cursor)}"
                              f"{'' if real_click else ' (not sent: realClick=false)'}",
                              flush=True)
                        last_click_ui = now_ms
                    if exit_match:
                        put(view, "EXIT GESTURE - leaving pointer", 88, (0, 0, 255), 0.6)
                    draw_pointer(view, pointer, now_ms, last_click_ui)
                    norm_for_gestures = None  # gestures paused in pointer mode
                else:
                    norm_for_gestures = norm

                if norm_for_gestures is not None:
                    if palm:
                        put(view, "OPEN PALM (arm)", 88, (255, 255, 0), 0.6)
                    elif fist:
                        put(view, "CLOSED FIST (disarm)", 88, (0, 0, 255), 0.6)
                    elif pointing or two:
                        what = "POINTING (pointer)" if pointing else "TWO FINGERS (scroll)"
                        msg = (f"{what} - hold" if trigger.armed or not pointer.require_armed
                               else f"{what} - arm first with open palm")
                        put(view, msg, 88, (255, 0, 255), 0.6)
                    elif store.gestures:
                        best, d, runner = store.match(norm)
                        runner_txt = f"  next {runner:.2f}" if runner is not None else ""
                        if best:
                            conf = max(0.0, 1 - d / store.match_threshold)
                            put(view, f'GESTURE: {best["name"]}  d={d:.2f}  '
                                      f'conf {conf:.0%}{runner_txt}', 88, (0, 255, 0), 0.6)
                        else:
                            put(view, f"no match  closest d={d:.2f}{runner_txt}",
                                88, (0, 140, 255), 0.6)
                if best:
                    last_match = (best, now_ms)
                    put(view, f"-> {actions.label(gmap.get(best['id']))}", 108,
                        (0, 255, 0), 0.5)
                # Bind menu open / pointer mode -> don't fire gesture actions
                fired = (None if binder.active or pointer.active
                         else trigger.update(best, palm, now_ms, fist=fist))
                if fired:
                    act = gmap.get(fired["id"])
                    ran = actions.run(act, dry_run)
                    print(f"FIRED: {fired['name']} -> {act or 'unbound'}"
                          f"{' (dry run)' if dry_run and ran else ''}", flush=True)
                if not pointer.active:
                    draw_trigger(view, trigger, now_ms, gmap)

            H = view.shape[0]
            for i, line in enumerate(recorder.lines(now_ms, norm is not None)):
                put(view, line, H - 30 + i * 18, (255, 255, 255), 0.5)
            blines = binder.lines(now_ms, gmap)
            for i, line in enumerate(blines):
                put(view, line, H - 66 - 18 * (len(blines) - 1) + i * 18,
                    (255, 255, 0), 0.45)
            names = ", ".join(
                f"{g['name']}->{actions.label(gmap.get(g['id']))}"
                for g in store.gestures) or "none yet"
            put(view, f"gestures: {names}", H - 50 if not blines else H - 30,
                (200, 200, 200))
            if not recorder.active and not binder.active:
                put(view, "click window: r record | b bind | z delete last | "
                          "p pointer on/off | q quit", H - 8, (200, 200, 200))
            put(view, f"last key {last_key}", 64, (160, 160, 160))

            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            if now - mark_t >= 1.0:
                calls_per_s = (pipe.model_calls - calls_mark) / (now - mark_t)
                calls_mark, mark_t = pipe.model_calls, now

            ram = proc.memory_info().rss / 1e6
            load = f"{tracker.load_ms:.0f} ms" if tracker.load_ms else "-"
            put(view, f"state {pipe.state}", 16, STATE_COLOR[pipe.state])
            put(view, f"FPS {fps:.0f}   model {tracker.last_ms:.1f} ms/call   "
                      f"{calls_per_s:.0f} calls/s", 32)
            put(view, f"RAM {ram:.0f} MB   model {'loaded' if tracker.loaded else 'unloaded'}"
                      f"   load {load}", 48)

            cv2.imshow("track", view)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            last_key = key  # shown on screen next frame (debug)
            consumed, changed = binder.key(key, now_ms, gmap)
            if changed:
                save_bindings()
            if consumed:
                continue
            if recorder.key(key, now_ms):
                continue  # typing a name: 'q' etc. are letters, not commands
            if key == ord("q"):
                break
            elif key == ord("u"):
                tracker.unload()
            elif key == ord("p"):
                pointer.enabled = not pointer.enabled
                if not pointer.enabled:
                    pointer.stop()
                binder.say(f"Pointer mode {'ON' if pointer.enabled else 'OFF'}", now_ms)
            elif key == ord("r"):
                recorder.start()
            elif key == ord("b"):
                if last_match and now_ms - last_match[1] < 3000:
                    binder.start(last_match[0])
                else:
                    binder.say("Show a recorded gesture first, then press b", now_ms)
            elif key == ord("z"):
                g = store.remove_last()
                if g and gmap.pop(g["id"], None):
                    save_bindings()
                recorder.say(f'Deleted "{g["name"]}"' if g else "Nothing to delete",
                             now_ms)
    finally:
        if gap_log:
            gap_log.close()
        tracker.unload()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
