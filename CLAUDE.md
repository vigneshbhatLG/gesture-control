# Gesture Control for webOS TV — Project Notes

Living doc: what we're building, what's decided, where we are. Update it at the
end of every working session. (Named CLAUDE.md so Claude Code loads it
automatically when working in this folder.)

Last updated: 2026-11-13 (steps 1–6 + R1 pointer mode done and confirmed live;
awaiting user's next requirements)

---

## Goal

Camera-based hand gesture control for LG webOS TVs.

- Works with a **built-in TV camera** (future models) or an **external USB camera**
  (current TVs without one).
- Turned **on/off from the official Settings app** (Enact JS).
- **Fully user-customizable**: users can record *any* gesture and bind it to
  *any* TV control (like rebinding keys in a game). Built-in gestures are just
  default presets.

## Working style

- The user wants to be **guided step by step**: explain *why*, keep steps small,
  user runs each step and sends screenshots before moving on.
- Prototype on laptop first, port to TV later, Settings app integration last.

---

## Plan / phases

| Phase | Where | Status |
|---|---|---|
| 1. Prototype | Laptop, Python | **In progress** |
| 2. Port | Native C++ service on webOS | Later |
| 3. Integrate | Official Settings app (Enact) via settingsservice | Last |

### Porting rule

Detection core uses only OpenCV/numpy-style calls that exist 1:1 in C++.
Camera access and settings sit behind small interfaces so only those swap:

| Layer | Prototype | TV |
|---|---|---|
| Camera | `core/camera.py` (`cv2.VideoCapture`) | `com.webos.service.camera` (built-in or USB) |
| Settings | `config.json` | `com.webos.settingsservice`, category `gesture`, **same key names** |
| Custom gesture templates | `gestures.json` (planned) | DB8 / service data dir on writable partition |
| Actions | keyboard keys (planned) | Luna key / app events |

---

## Decisions log

- **2026-11-13** — Started with classical CV (no ML): fixed ROI "gesture zone",
  background difference + YCrCb skin mask, contour = hand.
- **2026-11-13** — Classical CV hit its limits on the real office setup (see
  "Lessons"). User is OK with AI. **Decision: switch to MediaPipe Hands
  (pretrained, ~1–2M params, we train nothing)** for 21 hand landmarks.
  Keep the classical version in the repo for comparison.
- **2026-11-13** — **Custom gestures via few-shot template matching**, not
  neural-net training: user performs gesture 5×, store normalized landmarks,
  match by nearest neighbour with a rejection threshold. Pose gestures =
  landmark distance; motion gestures = DTW on trajectories. Optional tiny MLP
  (~6.5K params, 63→64→32→N) later if accuracy needs it.
- **2026-11-13** — OpenCV: plain `opencv-python` 5.0 lacks Haar cascades. In the
  venv we use **opencv-contrib-python 5.0** (MediaPipe needs it, and it has Haar).
- **2026-11-13** — Priority: **minimal space, max efficiency** (see Efficiency).

---

## Current state (prototype)

**Run everything with the project venv:** `.venv\Scripts\python <script>`
(setup: `python -m venv .venv`, `pip install -r requirements.txt`,
`python tools/get_models.py` — downloads hand model + face XML).

```
track.py               MediaPipe version (CURRENT): landmarks + lean pipeline + stats overlay
core/hands.py          HandTracker: MediaPipe HandLandmarker, VIDEO mode, lazy load/unload
core/pipeline.py       IDLE/SEARCH/TRACK state machine + MotionGate (64x48 frame diff)
core/gestures.py       normalize() + GestureStore: record checks, match, gestures.json
core/recorder.py       record-mode state machine (naming -> capture 5 samples -> check -> save)
core/trigger.py        DISARMED/ARMED/COOLDOWN: palm arms, hold fires, cooldown, release
core/actions.py        action registry (volume, mute, media, open YouTube) + repeat flag + TV mapping
core/binder.py         GTA-style bind menu: b -> 1..N bind, 0 unbind; same action moves
core/pointer.py        pointer mode (air mouse): OneEuro filter + activate/move/click/exit
core/mouse.py          laptop mouse output (ctypes); TV = webOS pointer injection later
gestures.json          user's recorded gestures (git-ignored)
models/                hand_landmarker.task + haarcascade XML (git-ignored; get_models.py)
logs/                  pointer_gaps*.csv thumb-gap logs, numbers only (git-ignored)
requirements.txt       mediapipe, opencv-contrib-python, numpy, psutil
tools/get_models.py    downloads both model files
config.json            settings; key names = future settingsservice keys
main.py                CLASSICAL version (kept for comparison): zone, contour, face box, debug mask
core/camera.py         camera wrapper (backend/index configurable, forces 320x240, mirror)
core/config.py         load/save config.json
core/segment.py        classical hand segmentation (diff / skin / both modes)
core/face.py           Haar face exclusion (so head isn't detected as hand)
tools/probe_cameras.py list camera indices + snapshot each (delete snapshots after)
tools/gesture_stats.py spread/separation of recorded gestures → calibrate thresholds
tools/gap_stats.py     thumb-gap distribution from logs/pointer_gaps.csv → calibrate click
tools/click_timing.py  per-click gap dip + interval → real taps vs jitter
tests/                 synthetic tests, one file per module; run: .venv\Scripts\python tests\run_all.py
```

**Tests** (2026-11-13: **104 pass**, 1 known limitation; exit code 0 = all good):

| File | Covers | Count |
|---|---|---|
| test_pointer.py | One Euro filter, ☝️ activation + arming, motion, thumb-tap click, jitter, adaptive thresholds, exits, ✌️ joystick scroll | 44 |
| test_trigger.py | palm arm, hold-to-fire, cooldown, repeat, dropout, timeout, fist disarm | 22 |
| test_gestures.py | normalisation, matching, record checks (incl. reserved ✌️), persistence, recorder keys | 16 |
| test_actions.py | bind menu (bind/move/unbind), per-action repeat, dispatch | 12 |
| test_face.py | Haar cascade + face exclusion (classical) | 6 |
| test_segment.py | classical segmentation + auto-exposure compensation | 4 (+1 known: +120 brightness saturation) |

They use fake hands / synthetic images — no camera needed. Run them after every change.
`run_all.py` counts a crashing script as a failure.

`track.py` keys: `q` quit, `u` unload model, `r` record gesture (type name, Enter,
hold gesture for 5 samples), `z` delete last gesture.

**Gesture matching details:** normalise = wrist to origin, scale by wrist→middle
knuckle (palm length), x × aspect, left hand mirrored (`mirrorLeftHand`).
**Rotation not normalised** (👍 ≠ 👎). Distance = RMS of 21 point distances in
palm lengths. Thresholds (config `gestures`): match ≤ 0.25, consistency spread ≤ 0.2,
distinct ≥ 0.3. Synthetic: same pose + noise ≈ 0.08, one finger different ≈ 0.41.
**Live data 2026-11-13** (`tools/gesture_stats.py`): 👍 "volume up" spread 0.02 (max 0.04),
👎 "volume down" spread 0.09 (max 0.16); separation 0.97. Live match d=0.07 vs next 1.01.
Thresholds fit so far; still to test with **similar** gestures (one finger apart, e.g. ✌️ vs 🤘).
2 gestures × 5 samples = 11.4 KB JSON (TV float16 binary: 840 B).

`main.py` keys: `q` quit, `b` relearn background, `m` cycle mode, `s` save
threshold/mode. Slider on mask window = diff threshold.

**Status of classical version:** finds the hand but unreliable — superseded by
the MediaPipe plan. Live test (2026-11-13): face exclusion **works**; `skin` cue was
cleanest in office light; `diff` still inverts under auto-exposure.

### Environment (user's laptop)

- Windows 11, Python **3.14**. MediaPipe **1.2.3 supports 3.14** — no extra Python needed.
- Project **venv `.venv`** has mediapipe 1.2.3 + **opencv-contrib-python 5.0** (has Haar,
  needed by MediaPipe). Never also install `opencv-python` there — both provide `cv2`.
- Global (non-venv) Python has opencv-python 4.14 from the earlier classical step.
- Webcam: **DirectShow index 1** (~28–31 FPS). DirectShow **index 0 is a virtual
  camera showing an "LG Electronics" placeholder** — not the real cam.
  MSMF index 0 is also the real cam (~24 FPS).
- **Handedness**: with our mirrored frames MediaPipe's Left/Right came out swapped →
  `hands.swapHandedness: true` in config. Re-check on the TV camera (may be unmirrored).
- OpenCV windows only get key presses when **focused** (user must click the window).
  `track.py` shows `last key <code>` for debugging; Enter arrives as 13 here.
- Overlay text uses a 1px drop shadow, not a thick outline (thick outline drifts:
  OpenCV glyph spacing depends on thickness).
- **OpenCV 5 contrib (venv) ships `CascadeClassifier` but NOT the Haar XML data** → classical
  `main.py` crashed from the venv. Fixed: `tools/get_models.py` also downloads
  `haarcascade_frontalface_default.xml` into `models/`; `core/face.py` looks there first.
- **Don't edit files with PowerShell `Get-Content`/`Set-Content`** (5.1 reads UTF-8 as ANSI
  → emoji in comments became mojibake + BOM added; happened to core/pointer.py, repaired).
  Use the Edit tool or Python.
- **Only one process can use the webcam.** If `main.py` (or Teams) is still running,
  `track.py` gets black frames at ~1 FPS. Check with
  `Get-CimInstance Win32_Process -Filter "Name like 'python%'"`.
- Driver ignores requested resolution (gives 640x360/480) → resized in software.
- Office: bright ceiling lights, beige walls/sweater (close to skin tone).

### Lessons from classical CV (why we're switching)

1. **Auto-exposure** shifts the whole image when anything moves → everything
   looks "changed". Fixed partly with median brightness compensation +
   "lighting change" guard (>70% of zone changed → relearn), but bright ceiling
   lights saturate non-uniformly and still leak through.
2. **Head/arm look identical to a hand** (moving, skin-coloured blob) → needed
   face detection to exclude the head.
3. Locking webcam exposure via OpenCV was inconclusive (black frames, cam was
   probably in use). Don't rely on it — TV USB cams vary anyway.

---

## Settings design (for Enact / settingsservice)

```json
{
  "gestureEnabled": true,
  "gestureSensitivity": 3,
  "gestureCooldownMs": 500,
  "gestureMapVersion": 1,
  "gestureMap": { "<gestureId>": "<actionId>" }
}
```

- Stored **gesture → action** (detector looks up by gesture); UI shows it by action.
- Gesture service **subscribes** (`getSystemSettings`, `subscribe: true`) → changes
  apply live, no restart.
- Custom gesture templates (bigger) live in **DB8**; `gestureMap` references them by id.
- Template record: `{id, name, type: "pose"|"motion", action, samples: [[63 floats], ...]}`.

### Remapping rules

1. One gesture → one action; conflict prompts "already used for X — swap?"
2. **Activation gestures** are built-in: hold open palm = arm, hold closed fist = disarm.
   Not remappable, and custom gestures can't look like them.
3. **Remote can always turn gestures off** — never gesture-only.
4. Reset to defaults always available (also what factory reset gives).
5. Actions may be unassigned.
6. "Perform the gesture now" capture flow needs a capture mode in the service
   (streamed to Settings via Luna subscription).

### Recording checks (custom gestures)

- Consistency: the 5 samples must resemble each other.
- Distinctiveness: reject/warn if too close to an existing gesture.
- Too generic: reject relaxed/neutral hands (false triggers).

---

## Efficiency: minimal space, max efficiency (user priority, 2026-11-13)

**Targets (unverified — measure on prototype):**

| | Target |
|---|---|
| Flash | **~1.5–3 MB** (if TFLite already in firmware; +1–3 MB if not) |
| RAM active | **~5–10 MB** |
| RAM idle | **~1 MB** (motion check only, models unloaded) |
| CPU idle | **~0%** |

**How:**
1. Ship only the **two MediaPipe model files** (palm detector + hand landmark,
   **lite** variants, **int8** quantized, ~1–2 MB) run via **TFLite** — not the
   MediaPipe framework.
2. **No OpenCV on TV**: crop/resize/rotate in plain C++; use camera hardware
   scaler for small frames.
3. **No face model** (MediaPipe finds hands directly).
4. **Pipeline: idle → motion → track → lost**
   - Idle: 64×48 gray frame diff, no model loaded.
   - Motion: palm detector once.
   - Track: landmark model only, on crop around last position.
   - Lost: palm detector again → idle after a few seconds.
5. **10–15 FPS** requested straight from the camera.
6. **Templates**: x,y only (z is noisy) = 42 values, **float16** → ~84 B/sample.
   Pose gesture (5 samples) ~420 B; motion gesture (resampled to 16 frames)
   ~7 KB; 50 gestures < 400 KB.
7. **On-demand service**: not running unless enabled **and** camera present;
   lazy-load models on first motion, unload after idle timeout.

**Measured on laptop (2026-11-13, MediaPipe framework, float16 "full" models):**

| | Value |
|---|---|
| Model bundle | 7.8 MB (hand_detector 2.34 MB + hand_landmarks 5.48 MB) |
| Inference | ~9 ms/call (no hand, palm detector), ~12 ms first call |
| Model load | ~3.6 s (cold, framework init) |
| RAM added by loading | ~55 MB (framework overhead >> model size → confirms: use bare TFLite on TV) |
| Whole `track.py` process | ~174 MB RSS (Python + OpenCV + MediaPipe + GUI) |
| Live, no hand (search 5/s) | 26 FPS camera, 24 model calls in 5 s, ~11 ms/call |
| Live, TRACK (user test) | 30 FPS, ~16 ms/call, 29 calls/s, conf 0.99, works with hand over face |
| Model reload (warm) | **58 ms** (cold first load ~2.8 s) → unload-when-idle is cheap |
| Pipeline (synthetic) | still: 0 calls/s · motion no hand: 5/s · hand: every frame · idle 3 s · unload 60 s |

Note: the MediaPipe Tasks bundle has no "lite" option; lite/int8 models are a later step.

**Rules:** never write camera frames to storage (privacy + eMMC wear). Models in
read-only system partition (firmware/OTA); user data on writable partition
(wiped by factory reset). Validate int8 + lite accuracy loss on prototype.

## Open questions for LG teams

- Platform/BSP: partition layout, firmware size budget, where system services install,
  **is TFLite already in firmware (biggest size unknown)**, NPU availability on target SoCs,
  can the camera deliver low-res / low-FPS frames directly (hardware scaler).
- Camera service: USB camera plug/unplug events, built-in vs external API.
- settingsservice: registering category `gesture`; can it store nested JSON objects
  (else store `gestureMap` as a JSON string).
- Legal/privacy: camera feature sign-off; **hand landmarks may count as biometric data**;
  MediaPipe (Apache 2.0) / model licences for shipping.

---

## Next steps

1. ~~venv + MediaPipe~~ done (2026-11-13).
2. ~~Hand tracking + idle/search/track pipeline + stats~~ **done, confirmed live** (2026-11-13).
3. ~~Record mode~~ **done, works live** 2026-11-13 (👍/👎 recorded + matched).
4. ~~Match mode + checks~~ built (same). Then: calibrate thresholds from live distances;
   "too generic" check (relaxed hand) still TODO; motion gestures (DTW) after.
   User recorded 3rd gesture "open youtube" (sep 0.82). One-finger-apart pair still untested.
   "volume up"/"open youtube" spread only 0.02 (hand held very still) → may need re-record.
5. ~~Trigger~~ **done, works live** 2026-11-13 (palm arms; "open youtube" ×2 and
   "volume up" ×2 fired correctly, model 9.5 ms/call).
   Activation = **built-in open palm** (rule `is_open_palm`: tips further from wrist than
   middle joints ×1.15, thumb tip >0.5 palm from index knuckle); custom gestures that look
   like open palm are rejected. Config `trigger`: armHoldMs 1000, armTimeoutMs 30000,
   holdMs 500, dropoutMs 150, repeatWhileHeld false; cooldown = `gestureCooldownMs`.
   Fired events print to console + "FIRED" on screen.
   **User request 2026-11-13: closed fist disarms.** Built-in `is_fist`: 4 fingertips
   closer to wrist than middle joints AND thumb tip < 0.4 palm from knuckles/fingers
   (thumb check keeps 👍/👎 from counting — their thumb gap is 0.73–1.19). Hold 500 ms
   (`disarmHoldMs`) disarms from ARMED or COOLDOWN; fist never fires as a gesture;
   fist-like recordings rejected. With explicit disarm, `armTimeoutMs` raised 5 s → 30 s
   (kept as safety net). `tools/gesture_stats.py` reports palm/fist conflicts.
6. ~~Action layer~~ **done, works live** 2026-11-13 (palm arm → 👍 real volume up with
   repeat → fist disarm → 👍 ignored; user also rebound via `b` menu):
   `gestureMap` = gesture **id** → action id (bindings NOT stored in gestures.json —
   templates and bindings separate, like DB8 vs settingsservice). Laptop actions =
   Windows media keys via ctypes `keybd_event` + browser for YouTube. Per-action repeat
   (volume repeats every `trigger.repeatMs` 250 ms; others once per hold). `b` bind menu.
   **Live reload** of config.json (mtime checked 1×/s; bad JSON keeps old settings).
   `gestureEnabled: false` → model unloaded, "GESTURE CONTROL OFF". `actions.dryRun`.
   Current bindings (end of 2026-11-13, user's own via `b` menu): volume up (👍)→volumeUp,
   volume down (👎)→volumeDown, "magic hands exit" (g_6efb5e, fingers curled + thumb out,
   spread 0.02)→exitPointer. 🤘 "magic hand exit" (g_927d0e) now **unbound**.
   "open youtube" was ✌️ → deleted for R2; needs re-recording as another gesture.
   Not done: "perform gesture now" capture flow for the Settings UI.
7. ~~Stats overlay~~ done in `track.py` (state, FPS, ms/call, calls/s, RAM, load time).
8. Run **int8 .tflite models directly via TFLite runtime** (like the TV will) and
   measure size / RAM / ms per frame vs. targets above.

### Where we left off (2026-11-13)

**Laptop prototype is feature-complete for pose gestures + pointer mode**, all confirmed
live: record any gesture, bind to any action, ✋ arm / ✊ disarm, ☝️ air-mouse with
thumb-tap click (real clicks ON), on/off + live reload.

**User is sending additional requirements one by one** (R1 so far) — add each here,
check impact on what's built, agree order before coding.

#### Built-in poses (reserved, not recordable as custom gestures)

| Pose | Rule | Does |
|---|---|---|
| ✋ open palm, hold 1 s | `is_open_palm` | **arm** gesture control |
| ✊ fist, hold 0.5 s | `is_fist` (thumb tucked) | **disarm** (also exits pointer) |
| ☝️ pointing, hold 0.3 s, **armed only** | `is_pointing` (thumb ignored) | **pointer mode** (move cursor, thumb-tap click) |
| ✌️ two fingers, **armed only** | `is_two_fingers` (index+middle up, ring+pinky curled, thumb ignored) | **scroll** (joystick) |

#### R1 (2026-11-13): Pointer mode ("air mouse", like Magic Remote) — DONE, works live

**How it works now:**
- ✋ arm → ☝️ hold 0.3 s → real mouse cursor follows the **index knuckle** (landmark 5),
  relative motion from where the cursor already is, One Euro smoothing,
  gain = 1 + 0.5 × `gestureSensitivity`.
- **Click** = thumb tap: stick thumb out (👉), tap it in against the middle finger.
  Thumb gap = median of 5 frames; press must last 2 frames; **adaptive thresholds**
  (press 50 % / release 75 % of the learned "thumb out" level, last ~3 s); 300 ms
  debounce; cursor frozen while pressed. `realClick: true` sends the real left click.
- Cursor moves **only while pointing**; any other shape freezes it and after 0.3 s exits.
- **Exit**: stop pointing 0.3 s (e.g. open palm), 🤘 "magic hand exit" (bound to
  `exitPointer`) 0.6 s, hand lost 0.6 s, or ✊ 0.6 s (✊ also disarms).
- Pointer use keeps arming alive; custom gestures are paused while the pointer is active.
- Keys: `p` toggles pointer mode. Config: `pointer` section. Logs:
  `logs/pointer_gaps.csv` (numbers only) → `tools/gap_stats.py`, `tools/click_timing.py`.
- Tests: `tests/test_pointer.py` 28/28 (filter, activation, arming, move, click, jitter,
  adaptive, exits). All other suites pass.
- **Not yet**: drag / long-press; resolution switch for TV distance (2–3 m, a finger is
  only a few px at 320×240); TV pointer injection API (ask platform team).

#### R2 (2026-11-13): Scrolling up / down / sideways — built, waiting on live test

**User decisions**: ✌️ **reserved** for scroll (not context-based) → user's ✌️ gesture
"open youtube" (g_dd9c5e, 5/5 samples were ✌️) **deleted with its binding; user to
re-record "open youtube" as a different gesture**. Scroll style: **joystick**.

**How it works:**
- In pointer mode (or straight from armed): ✌️ → **SCROLL** sub-mode. Where ✌️ starts =
  joystick centre. Move hand away → page keeps scrolling that way; speed rises from 0 at
  the deadzone (0.15 palm lengths) to `maxNotchesPerSec` 12 at `maxOffset` 0.8 palm
  lengths; back to centre = stop. **Axis lock**: only the dominant direction scrolls.
- Offsets in **palm lengths** (wrist→middle knuckle) → same feel near or far from camera.
- Cursor frozen and clicks off while scrolling. ☝️ again → MOVE (re-anchored, no jump,
  thumb must open before next click). Neither ☝️ nor ✌️ for 0.3 s → pointer exits.
- Laptop output: `Mouse.scroll` → `mouse_event` WHEEL / HWHEEL (120 = 1 notch; smaller
  deltas = smooth). TV: Magic Remote wheel / scroll injection — ask platform team.
- Overlay: joystick (deadzone disc, full-speed ring, hand dot) + "UP 6/s" etc.
- Config: `pointer.scroll` (enabled, deadzone, maxOffset, maxNotchesPerSec, axisLock,
  minCutoff, beta). Tests: 16 scroll tests in `test_pointer.py` (direction, speed, axis
  lock, deadzone, distance independence, back to cursor). Total suite: 104 pass.

**History of R1 (how we got here):**

Original design notes:
- **Activate**: ☝️ index up, others curled → cursor shows. Hand drop / ✊ → hides.
  ☝️ becomes a **reserved built-in pose** (like ✋ ✊), not recordable.
- **Move**: anchor = **index knuckle (MCP, landmark 5)**, not fingertip (tip moves when
  clicking). Virtual "touchpad" box in the frame mapped to full screen; box size from
  `gestureSensitivity`. **One Euro filter** for smoothing (steady at rest, low lag moving).
- **Click** options: A thumb-tap against middle finger side (recommended), B pinch
  (thumb+index tips), C air tap (needs depth z, unreliable). Hold → drag/long-press later.
- **Laptop**: real mouse via ctypes `SetCursorPos` + `mouse_event`. **TV**: webOS pointer
  injection — confirm API with platform team.
- **Risk**: at 2–3 m a finger is a few px at 320×240 → pointer mode likely needs 640×480
  while active (efficiency trade-off; switch resolution only in pointer mode).
- **User decided**: **built-in ☝️ rule** (not a bindable action) + **thumb-tap click (A)**.
  User had recorded ☝️ as custom "magic hands" (spread 0.19, max 0.42 — inconsistent);
  removed as agreed since ☝️ is now reserved.
- **Built 2026-11-13, synthetic-tested (17/17), waiting on live test**:
  `core/pointer.py` (OneEuro + Pointer state machine, pure logic), `core/mouse.py`
  (ctypes SetCursorPos/mouse_event, DPI-aware), `is_pointing` in gestures.py (index tip
  further than PIP ×1.15, middle/ring/pinky curled; thumb ignored) — real data: "magic
  hands" 5/5 pointing, other gestures 0/5. ☝️ alone activates (no arming needed), hold
  300 ms. Relative motion from current OS cursor, gain = 1 + 0.5 × gestureSensitivity.
  Click = thumb gap < 0.35 after > 0.55 (hysteresis), 300 ms debounce, cursor frozen while
  pressed. Exit: hand lost 600 ms or fist 600 ms. Gestures paused while pointer active.
  Config `pointer` (enabled, moveRealCursor, thresholds, filter). Key `p` toggles.
  Overlay shows thumb-gap meter with press/release marks → calibrate thresholds live.
  **Watch**: user's natural ☝️ has thumb gap 0.28–0.52 (half-tucked) → may need to point
  with thumb out ("gun" shape) so the tap has room; tune pressGap/releaseGap from live data.
- **Live test 1 (2026-11-13)**: cursor moved fine, BUT **~85 false clicks** (3–4/s at the
  same spot) → raw thumb landmark jitters in pointing pose. Fixes: **`realClick: false`**
  (clicks shown/logged, real mouse button NOT pressed until calibrated), gap = median of
  5 frames, press must last `pressFrames` 2 frames. Gap values logged to
  `logs/pointer_gaps.csv` (`logGaps`, numbers only) for calibration. Synthetic jitter
  test: 0 clicks.
- User recorded 🤘 "magic hand exit" (spread 0.09, good) → new bindable action
  **`exitPointer`** (only matched while pointer active; hold 600 ms). Bound by default.
- **Live bug 2 (2026-11-13)**: cursor moved with an **open palm** (pointer stayed active
  for any shape). Fix: cursor moves **only while `is_pointing`** (thumb ignored, so clicks
  still work); not pointing for `notPointingExitMs` 300 → exit; re-anchor on resume (no
  jump). Also fixed: `last_seen` not set on activation. Tests 22/22.
- **Gap data** (`tools/gap_stats.py`, 2557 frames / 109 s, after median fix): clearly
  **bimodal** — tucked 0.10–0.35, out 0.70–0.85; raw jitter median 0.009, p95 0.115.
  28 clicks (15/min), down from ~85. Thresholds widened to press < **0.30**, release >
  **0.60** (in the valley between the modes).
- **Live bug 3 (2026-11-13): "click is failing"** — 2nd session's thumb-out level was
  0.55–0.75 (vs 0.70–0.85 before), below the fixed release 0.60 → never released → taps
  ignored. Fix: **adaptive thresholds** (`pointer.adaptive`): open_ref = 85th pct of last
  90 frames' gap (floor 0.5); press = 0.5×, release = 0.75× open_ref. Learning frozen
  only while a click is held (`click_held`; freezing on `pressed` deadlocked — caught by
  tests). Synthetic: out 0.58 → 5/5 (fixed: 0/5); out 0.80 → 5/5; long hold → 1 click.
- **User rule (2026-11-13): pointer needs arming too** — `pointer.requireArmed: true`:
  ☝️ starts pointer only when trigger is ARMED (open palm). Pointer use keeps arming alive
  (`trigger.keep_alive`); fist exit from pointer also disarms (`trigger.disarm`).
  Overlay says "POINTING - arm first with open palm" when disarmed.
- **Verified 2026-11-13** (`tools/click_timing.py`): 30 clicks, all real taps — deep dips
  (gap ~0.5 → 0.10–0.23, depth 0.21–0.58), ~1 s apart, no 300 ms jitter bursts. User's
  "click not working" was just `realClick: false` (clicks detected but not sent).
  → **`realClick` turned ON**; user confirmed ("cool"). Note: many clicks happened at
  screen top-right (cursor clamped at edge) = window close buttons → careful.
- Superseded by the above: "☝️ alone activates" (now needs arming), fixed thresholds
  0.35/0.55 then 0.30/0.60 (now adaptive), "stays active for any hand shape" (now only
  while pointing).

Remaining roadmap (my recommended order):
0. **Live-test R2 scroll** (last session: 49 real clicks + exit gesture used; scroll events
   aren't printed, so unknown whether scroll was tried — user said "good" and stopped).
   Then next requirement from the user (R3…). Tracker was stopped at end of session.
1. **TV footprint**: item 8 above (int8 + bare TFLite) — real numbers for platform team.
2. **Robustness**: TV distance 2–3 m (may need higher camera res), one-finger-apart pair
   (✌️ vs 🤘), "too generic" check, re-record gestures with too-tight spread.
3. **Motion gestures** (DTW): circle, wave, swipe.
4. **Porting prep**: spec for the C++ service + Enact Settings screen.
5. **"Perform gesture now"** capture flow for the Settings UI.
6. Pointer extras: drag / long-press, higher resolution while pointing.

Housekeeping: git repo has **no commits yet** — offered a first commit (user to confirm).
