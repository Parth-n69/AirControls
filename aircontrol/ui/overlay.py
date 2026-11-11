"""HUD and landmark overlay renderer for AirControl.

Draws real-time computer vision telemetry onto camera frames:
- Hand landmark skeleton and joint highlights
- Active-region interaction boundary box with corner reticles
- System state badges (IDLE / ARMED / ACTIVE) and gesture indicators
- Fist hold arming progress bar / countdown
- Performance telemetry (Camera FPS & Tracking latency)
- Low-light warning banner
"""

from __future__ import annotations
from typing import Tuple, Optional
import cv2
import numpy as np

from aircontrol.core.tracker import (
    HandData,
    HAND_CONNECTIONS,
    INDEX_TIP,
    THUMB_TIP,
    WRIST,
)
from aircontrol.core.config import ActiveRegionConfig


# Styling color palette (BGR)
COLOR_CYAN = (245, 200, 30)         # Hand bones
COLOR_MAGENTA = (200, 50, 240)      # Joint landmarks
COLOR_ACCENT_GREEN = (50, 230, 80)  # Active state / index tip
COLOR_AMBER = (30, 180, 255)        # Armed / warning state
COLOR_CRIMSON = (40, 40, 220)       # Idle / off state
COLOR_DARK_BG = (25, 25, 25)        # Badge background
COLOR_TEXT = (250, 250, 250)        # Label text
COLOR_BOX = (200, 200, 200)         # Active zone border


class OverlayRenderer:
    """Renders visual HUD elements and hand overlays directly onto OpenCV frames."""

    def __init__(self) -> None:
        pass

    def draw_active_region_box(
        self,
        frame: np.ndarray,
        active_box: ActiveRegionConfig,
        flip_horizontal: bool = True,
    ) -> None:
        """Draw active tracking region rectangle with high-tech corner accents."""
        h, w = frame.shape[:2]
        min_x, min_y, max_x, max_y = active_box.as_tuple()

        # In un-mirrored preview, horizontal coordinates match directly;
        # in mirrored preview, horizontal orientation matches user perspective
        px1 = int(min_x * w)
        py1 = int(min_y * h)
        px2 = int(max_x * w)
        py2 = int(max_y * h)

        # Draw semi-transparent bounding zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_BOX, 1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw tech corner markers (20px brackets)
        corner_len = 20
        thick = 2
        # Top-Left
        cv2.line(frame, (px1, py1), (px1 + corner_len, py1), COLOR_CYAN, thick)
        cv2.line(frame, (px1, py1), (px1, py1 + corner_len), COLOR_CYAN, thick)
        # Top-Right
        cv2.line(frame, (px2, py1), (px2 - corner_len, py1), COLOR_CYAN, thick)
        cv2.line(frame, (px2, py1), (px2, py1 + corner_len), COLOR_CYAN, thick)
        # Bottom-Left
        cv2.line(frame, (px1, py2), (px1 + corner_len, py2), COLOR_CYAN, thick)
        cv2.line(frame, (px1, py2), (px1, py2 - corner_len), COLOR_CYAN, thick)
        # Bottom-Right
        cv2.line(frame, (px2, py2), (px2 - corner_len, py2), COLOR_CYAN, thick)
        cv2.line(frame, (px2, py2), (px2, py2 - corner_len), COLOR_CYAN, thick)

        # Label
        cv2.putText(
            frame,
            "ACTIVE CONTROL ZONE",
            (px1 + 8, py1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def draw_hand_skeleton(
        self,
        frame: np.ndarray,
        hand: HandData,
        is_dominant: bool = True,
    ) -> None:
        """Render hand landmarks and bone connections."""
        bone_color = COLOR_CYAN if is_dominant else (140, 140, 140)
        joint_color = COLOR_MAGENTA if is_dominant else (100, 100, 100)

        # Draw bone connections
        for idx1, idx2 in HAND_CONNECTIONS:
            pt1 = (hand.landmarks[idx1].px, hand.landmarks[idx1].py)
            pt2 = (hand.landmarks[idx2].px, hand.landmarks[idx2].py)
            cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        # Draw joints
        for i, lm in enumerate(hand.landmarks):
            pt = (lm.px, lm.py)
            radius = 3
            color = joint_color

            # Specially highlight fingertips
            if i in (INDEX_TIP, THUMB_TIP):
                radius = 5
                color = COLOR_ACCENT_GREEN if is_dominant else joint_color

            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

        # Draw palm center
        cv2.circle(frame, hand.palm_center_px, 4, COLOR_AMBER, -1, cv2.LINE_AA)

    def draw_hud(
        self,
        frame: np.ndarray,
        state_label: str,
        gesture_label: str,
        fps: float,
        latency_ms: float,
        hold_progress: float = 0.0,
        is_low_light: bool = False,
    ) -> None:
        """Draw complete HUD badges, state indicator, and performance metrics."""
        h, w = frame.shape[:2]

        # 1. State Badge (Top Left)
        if state_label.upper() == "ACTIVE":
            badge_bg = COLOR_ACCENT_GREEN
            badge_text = "AIRCONTROL: ACTIVE"
            text_color = (0, 0, 0)
        elif state_label.upper() == "ARMED":
            badge_bg = COLOR_AMBER
            badge_text = "ARMING..."
            text_color = (0, 0, 0)
        else:
            badge_bg = COLOR_DARK_BG
            badge_text = "STANDBY (Hold Fist 1s)"
            text_color = (200, 200, 200)

        # Draw state pill
        pill_w = 230
        pill_h = 32
        cv2.rectangle(frame, (16, 16), (16 + pill_w, 16 + pill_h), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (16, 16), (22, 16 + pill_h), badge_bg, -1)
        cv2.putText(
            frame,
            badge_text,
            (32, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color if state_label.upper() != "IDLE" else (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # 2. Fist hold progress bar (if arming / holding fist)
        if 0.0 < hold_progress <= 1.0:
            bar_w = pill_w
            progress_w = int(bar_w * hold_progress)
            cv2.rectangle(frame, (16, 16 + pill_h + 4), (16 + bar_w, 16 + pill_h + 9), (50, 50, 50), -1)
            cv2.rectangle(frame, (16, 16 + pill_h + 4), (16 + progress_w, 16 + pill_h + 9), COLOR_AMBER, -1)

        # 3. Gesture Label Badge
        if gesture_label and gesture_label != "NONE":
            g_pill_w = 200
            cv2.rectangle(frame, (16, 62), (16 + g_pill_w, 90), COLOR_DARK_BG, -1)
            cv2.putText(
                frame,
                f"Gesture: {gesture_label}",
                (24, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                COLOR_CYAN,
                1,
                cv2.LINE_AA,
            )

        # 4. FPS & Latency (Top Right)
        fps_text = f"{fps:.1f} FPS | {latency_ms:.1f}ms"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        fps_x = w - tw - 24
        cv2.rectangle(frame, (fps_x - 8, 16), (w - 16, 42), COLOR_DARK_BG, -1)
        cv2.putText(
            frame,
            fps_text,
            (fps_x, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # 5. Low-Light Warning (Bottom Center)
        if is_low_light:
            warn_text = "LOW LIGHT WARNING: Detection may be impaired"
            (ww, _), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            wx = max(10, (w - ww) // 2)
            cv2.rectangle(frame, (wx - 10, h - 38), (wx + ww + 10, h - 12), COLOR_DARK_BG, -1)
            cv2.rectangle(frame, (wx - 10, h - 38), (wx + ww + 10, h - 12), COLOR_AMBER, 1)
            cv2.putText(
                frame,
                warn_text,
                (wx, h - 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_AMBER,
                1,
                cv2.LINE_AA,
            )

