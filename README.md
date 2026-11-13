# AirControl 🖐️

> **Production-Quality Touchless Desktop Control via Mid-Air Hand Gestures**

Control your computer seamlessly using mid-air hand gestures captured by your standard webcam. No mouse, no touch, zero specialized hardware required. Built in Python 3.10+ with OpenCV, MediaPipe, PySide6, PyAutoGUI, and pycaw.

---

## 📸 Gesture Cheat-Sheet

| Gesture | Hand Posture | Action Performed | Behavior & Safeguards |
|---|---|---|---|
| **Toggle Control ON / OFF** | ✊ **Fist (Held 1.0s)** | Toggle System State | Yellow on-screen countdown bar prevents accidental toggles. |
| **Move Cursor** | ☝️ **Index Finger Up** | Fluid Mouse Cursor Movement | Exponential Moving Average (EMA) smoothing + deadzone to eliminate jitter. |
| **Left Click** | 🤏 **Quick Pinch (Thumb + Index)** | Single Left Click | Triggers when pinch released within 250ms. |
| **Click & Drag** | 🤏 **Pinch and Hold (> 250ms)** | Mouse Down & Drag | Mouse down engages; drag items across screen; release to drop. |
| **Scroll Page** | ✌️ **Two Fingers Up + Move** | Vertical Page Scrolling | Moving hand UP scrolls up; moving hand DOWN scrolls down. |
| **Previous Slide / Page** | 🖐️ **Open Palm Swipe LEFT** | Left Arrow Key (`←`) | Direction-locked horizontal motion with 0.5s debounce cooldown. |
| **Next Slide / Page** | 🖐️ **Open Palm Swipe RIGHT** | Right Arrow Key (`→`) | Direction-locked horizontal motion with 0.5s debounce cooldown. |
| **Volume Up (+5%)** | 🖐️ **Open Palm Swipe UP** | Master Volume Step Up | Increases system volume by 5% per flick. |
| **Volume Down (-5%)** | 🖐️ **Open Palm Swipe DOWN** | Master Volume Step Down | Decreases system volume by 5% per flick. |
| **Continuous Volume Slider** | 🤟 **Three Fingers Up + Move** | Analog Master Volume (0-100%) | Hand height in active zone smoothly sets master audio volume. |

---

## ⚡ Key Engineering Highlights

- **>= 25 FPS on CPU-Only Hardware**: MediaPipe inference runs on a downscaled frame with `model_complexity=0` and XNNPACK CPU acceleration, keeping laptop CPU usage minimal.
- **Threaded Video Pipeline**: Dedicated camera capture thread drops stale frames atomically, ensuring zero lag between your hand motion and cursor movement.
- **Active-Region Bounding Box**: You don't need to stretch across the webcam field-of-view; only a comfortable centered box maps to your entire monitor resolution.
- **Scale-Invariant Gesture Recognition**: All geometric distances are normalized by hand span ($dist(\text{Wrist}, \text{Middle MCP})$) and auto-scaled via a built-in calibration wizard.
- **Strict Privacy Guarantee**: Diagnostic logs are recorded to `aircontrol.log`. **Zero camera frames, video clips, or biometric data are ever saved to disk or transmitted over any network.**
- **Robust Exception Handling**: Gracefully handles busy cameras, missing devices, low-light environments, and left-handed mode.

---

## 🚀 Quick Start Guide

### 1. Requirements & Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14 (64-bit)
- Standard USB or built-in webcam (or use the built-in **Default Virtual Camera** if no webcam is present!)
- Windows 10 / 11 (pycaw audio support)

### 2. Installation
```bash
# Clone or navigate to the repository
cd "c:\air motion"

# Install production dependencies
pip install -r requirements.txt
```

### 3. Launching AirControl
```bash
# Run desktop GUI
python -m aircontrol.app

# Or run the standalone Windows binary
.\dist\AirControl\AirControl.exe

# Or start directly in system tray
python -m aircontrol.app --tray
```

> **No Webcam? No Problem!**
> If you don't have a webcam connected or your laptop's camera shutter is closed, AirControl automatically switches to the **Default Virtual Camera**. You can immediately test cursor movement, clicking, scrolling, volume, and swipes using the interactive **Gesture Quick-Select Deck** right in the UI (`[🔄 Auto Cycle]`, `[☝️ Move Cursor]`, `[👌 Pinch Click]`, `[✌️ Scroll]`, `[↔️ Swipe Nav]`, `[🔊 Swipe Volume]`, `[✊ Fist Hold]`). Whenever a physical webcam is connected, click **"Try Physical Webcam"** to switch over instantly.

---

## 🛠️ Configuration (`config.json`)

All thresholds, active zone coordinates, and sensitivities are fully configurable in `config.json`:

```json
{
    "camera": {
        "camera_index": 0,
        "width": 640,
        "height": 480,
        "inference_width": 480,
        "inference_height": 270,
        "target_fps": 30,
        "flip_horizontal": true,
        "low_light_threshold": 30.0
    },
    "active_region": {
        "min_x": 0.20,
        "min_y": 0.20,
        "max_x": 0.80,
        "max_y": 0.80
    },
    "mouse": {
        "smoothing_alpha": 0.35,
        "deadzone_pixels": 3.0,
        "speed_multiplier": 1.0,
        "drag_hold_delay_s": 0.25,
        "scroll_factor": 1.5,
        "scroll_deadzone": 0.015
    },
    "gestures": {
        "fist_hold_duration_s": 1.0,
        "pinch_distance_ratio": 0.055,
        "pinch_release_ratio": 0.075,
        "swipe_min_distance": 0.15,
        "swipe_min_velocity": 0.45,
        "swipe_direction_ratio": 1.8,
        "swipe_cooldown_s": 0.50,
        "volume_step": 0.05,
        "consecutive_frames_required": 3
    },
    "preferences": {
        "dominant_hand": "Right",
        "calibration_scale": 1.0,
        "start_in_tray": false,
        "show_landmark_skeleton": true,
        "show_active_region_box": true,
        "show_hud_stats": true
    }
}
```

---

## 🧪 Running Automated Unit Tests

AirControl features an extensive automated test suite with synthetic landmark fixtures that test the entire gesture and action pipeline without needing a live webcam:

```bash
# Run all unit tests
pytest

# Run with verbose output
pytest -v
```

---

## 📦 Building Standalone Windows Executable (`.exe`)

To compile AirControl into a zero-dependency Windows `.exe` application:

```bash
python build_exe.py
```

The resulting distribution directory will be output at `dist/AirControl/AirControl.exe`.

---

## 📜 Architecture Overview

```
aircontrol/
├── app.py                      # Application bootstrap & Qt event loop
├── core/
│   ├── camera.py               # Threaded OpenCV capture, frame drop, low-light telemetry
│   ├── tracker.py              # MediaPipe Hands wrapper (Tasks API + XNNPACK)
│   ├── classifier.py           # Scale-invariant static pose classification
│   ├── swipe.py                # Trajectory analysis, velocity tracking, direction lock
│   ├── state.py                # FSM (IDLE/ARMED/ACTIVE), N-frame stabilizer, 1s hold timer
│   ├── dispatcher.py           # Central action engine routing gestures to OS dispatchers
│   └── config.py               # JSON configuration parser and schema validator
├── actions/
│   ├── mouse.py                # EMA cursor smoothing, active box mapping, drag & scroll
│   ├── keyboard.py             # Directional key triggers (Left/Right arrows)
│   └── volume.py               # Abstract VolumeController + Windows pycaw audio backend
├── ui/
│   ├── main_window.py          # Modern PySide6 desktop UI with live HUD overlay
│   ├── settings_dialog.py      # Sensitivity sliders and hardware preferences
│   ├── calibration_dialog.py   # Hand span measurement wizard
│   ├── overlay.py              # OpenCV HUD graphics and landmark skeleton renderer
│   └── tray_icon.py            # System tray icon with background minimize support
└── utils/
    ├── logger.py               # Rotating file diagnostics (zero frames recorded)
    └── math_helpers.py         # EMA filters, coordinate transforms, vector distances
```

---

## 📄 License
MIT License. Built for production touchless computer interaction.

