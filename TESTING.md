# AirControl: Manual Quality Assurance & Gesture Test Cases

This document outlines step-by-step manual test cases to verify all mid-air gesture interactions, error boundaries, and telemetry in **AirControl**.

---

## 1. Test Environment Setup

- **Hardware**: Windows 10/11 laptop or desktop equipped with a standard webcam (720p or 1080p).
- **Positioning**: Sit upright 0.5 to 1.0 meters in front of the camera with your hand clearly framed.
- **Launch Application**:
  ```bash
  python -m aircontrol.app
  ```
- **Initial Verification**:
  - Live video preview appears in the main window.
  - Telemetry shows `~30 FPS` with `< 20ms` tracking latency.
  - System State indicates `IDLE (Hold Fist 1s)`.

---

## 2. Core Gesture Test Cases

### TC-01: Toggle Control ON / OFF (Fist Hold 1.0s)
- **Objective**: Ensure the system does not act on accidental gestures until explicitly armed.
- **Action**:
  1. Clench your hand into a tight **FIST** in front of the camera.
  2. Observe the HUD: status badge changes to yellow `ARMING...` and the progress bar fills over 1.0 second.
  3. Keep holding until the bar completes.
- **Expected Outcome**:
  - State switches to vibrant green `AIRCONTROL: ACTIVE`.
  - Windows notification appears: `"Gesture Control is now ACTIVE"`.
  - Hold fist again for 1.0s $\rightarrow$ switches back to `IDLE` (Control OFF).

---

### TC-02: Cursor Movement (Index Finger Up / Point)
- **Objective**: Verify smooth cursor positioning with active-region mapping and jitter suppression.
- **Action**:
  1. Arm the system to `ACTIVE`.
  2. Extend only your **INDEX FINGER** upward (curling middle, ring, pinky).
  3. Move your fingertip slowly within the corner-accented **ACTIVE CONTROL ZONE**.
  4. Hold your hand completely still for 3 seconds.
- **Expected Outcome**:
  - Windows mouse cursor tracks your index fingertip across the entire desktop.
  - Exponential smoothing (EMA) provides fluid motion without lag.
  - When holding still, the deadzone filter eliminates cursor micro-shiver.

---

### TC-03: Left Click (Quick Pinch)
- **Objective**: Verify responsive clicking without accidental dragging.
- **Action**:
  1. Position the cursor over an icon, button, or link.
  2. Bring the tips of your **THUMB** and **INDEX FINGER** together rapidly into a pinch and immediately release ($< 250\text{ms}$).
- **Expected Outcome**:
  - Exactly one Windows OS left-click is executed.
  - Dispatched action card displays `"Quick pinch triggered: CLICK"`.

---

### TC-04: Click-and-Hold Drag (Pinch and Hold)
- **Objective**: Ensure stable drag-and-drop mechanics.
- **Action**:
  1. Move the cursor over a desktop icon or window title bar.
  2. Pinch your thumb and index finger together and **HOLD** for $> 300\text{ms}$.
  3. Move your hand across the active zone while maintaining the pinch.
  4. Release the pinch.
- **Expected Outcome**:
  - Mouse down engages once the hold delay expires (`"DRAG STARTED"`).
  - The window/icon follows the cursor smoothly.
  - Releasing fingers drops the object cleanly without an extra stray click.

---

### TC-05: Page Scrolling (Two Fingers Up)
- **Objective**: Verify bidirectional vertical scrolling.
- **Action**:
  1. Open a web browser or document.
  2. Extend both **INDEX and MIDDLE FINGERS** upward (peace/V-sign posture).
  3. Move your hand upward $\rightarrow$ verify page scrolls UP.
  4. Move your hand downward $\rightarrow$ verify page scrolls DOWN.
- **Expected Outcome**:
  - Smooth page scrolling proportional to hand movement speed.
  - Action card displays `"Scrolling: N clicks"`.

---

### TC-06: Directional Swipe Navigation (Left / Right Arrow)
- **Objective**: Verify single-trigger navigation swipes with direction lock and cooldown.
- **Action**:
  1. Open a PowerPoint slideshow, PDF reader, or photo viewer.
  2. Show a flat **OPEN PALM** towards the camera.
  3. Perform a brisk horizontal swipe to your **RIGHT** (approx. 20–30 cm).
  4. Perform a brisk horizontal swipe to your **LEFT**.
- **Expected Outcome**:
  - Swipe Right triggers the **Right Arrow** key (advances to next slide/photo).
  - Swipe Left triggers the **Left Arrow** key (returns to previous slide/photo).
  - Debounce cooldown prevents accidental multi-skips.

---

### TC-07: Stepped Volume Control (Swipe Up / Down)
- **Objective**: Verify audio step adjustments via vertical swipes.
- **Action**:
  1. Show an **OPEN PALM**.
  2. Flick your hand briskly **UPWARD** vertically.
  3. Flick your hand briskly **DOWNWARD** vertically.
- **Expected Outcome**:
  - Swipe Up increases master volume by exactly **5%** (e.g. 50% $\rightarrow$ 55%).
  - Swipe Down decreases master volume by exactly **5%** (e.g. 55% $\rightarrow$ 50%).
  - Diagonal motions do not trigger accidental volume changes (direction dominance lock).

---

### TC-08: Continuous Volume Slider (Three Fingers Up)
- **Objective**: Verify real-time analog volume adjustment.
- **Action**:
  1. Extend **INDEX, MIDDLE, and RING FINGERS** upward.
  2. Move your hand towards the top of the Active Zone $\rightarrow$ volume smoothly rises towards 100%.
  3. Move your hand towards the bottom of the Active Zone $\rightarrow$ volume smoothly lowers towards 0%.
- **Expected Outcome**:
  - Master volume tracks hand height in real time without stepped jumps.
  - Dispatched action card displays `"Continuous volume: X%"`.

---

## 3. Edge Cases & Safety Verifications

| Test Case | Procedure | Expected Result |
|---|---|---|
| **Low-Light Warning** | Cover webcam partially or dim room lights below threshold ($< 30$ luminance). | Yellow warning banner appears at bottom: `"LOW LIGHT WARNING: Detection may be impaired"`. App continues running safely. |
| **Multiple Hands** | Bring both hands into webcam frame simultaneously. | App filters and tracks strictly the designated **Dominant Hand** (configured in Settings as Right or Left). |
| **Camera Disconnect** | Unplug USB webcam or toggle camera privacy switch. | Live feed halts gracefully and displays clear guidance: `"Failed to read frame / Reconnecting..."`. App automatically resumes when camera returns. |
| **Hand Calibration Wizard** | Click `"Calibrate Hand Size"` in UI, show open palm, and allow 35 frames to sample. | Hand span is measured, calibration scale updates (e.g. `1.15x`), and is persisted to `config.json`. |
| **System Tray Backgrounding** | Close main window (`X`) or click `"Minimize to Tray"`. | Window hides to system tray; right-clicking tray icon allows toggling control or reopening UI. |

