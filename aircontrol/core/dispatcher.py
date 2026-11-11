"""Action Dispatcher orchestrating gestures, FSM state, and OS commands.

Maps stabilized gestures and dynamic swipes to mouse, keyboard, and volume actions:
- POINT             -> Cursor movement (EMA smoothed, active-region bounded)
- PINCH             -> Mouse Left Click / Pinch-and-Hold Drag
- TWO_FINGERS_UP    -> Page Scrolling
- THREE_FINGERS_UP  -> Continuous vertical volume slider
- SWIPE_LEFT        -> Left Arrow (Previous slide/page)
- SWIPE_RIGHT       -> Right Arrow (Next slide/page)
- SWIPE_UP          -> Volume Up (+5%)
- SWIPE_DOWN        -> Volume Down (-5%)
"""

from __future__ import annotations
import time
from typing import Optional, Tuple
from dataclasses import dataclass

from aircontrol.core.config import AppConfig
from aircontrol.core.tracker import HandData, INDEX_TIP
from aircontrol.core.classifier import StaticGesture, ClassificationResult
from aircontrol.core.swipe import SwipeDetector, SwipeEvent, SwipeDirection
from aircontrol.core.state import ControlStateMachine, StateSnapshot
from aircontrol.actions.mouse import MouseController
from aircontrol.actions.keyboard import KeyboardController
from aircontrol.actions.volume import VolumeController
from aircontrol.utils.math_helpers import clamp
from aircontrol.utils.logger import get_logger

logger = get_logger("dispatcher")


@dataclass
class DispatchResult:
    """Summary of actions executed in the current frame."""
    action_type: str
    detail: str
    cursor_pos: Optional[Tuple[int, int]] = None
    volume_level: Optional[float] = None


class ActionDispatcher:
    """Central engine translating computer vision events into OS interactions."""

    def __init__(
        self,
        config: AppConfig,
        state_machine: ControlStateMachine,
        swipe_detector: SwipeDetector,
        mouse_controller: MouseController,
        keyboard_controller: KeyboardController,
        volume_controller: VolumeController,
    ) -> None:
        self.config = config
        self.state_machine = state_machine
        self.swipe_detector = swipe_detector
        self.mouse = mouse_controller
        self.keyboard = keyboard_controller
        self.volume = volume_controller

        # 3-finger continuous volume slider EMA
        self._volume_slider_active = False

    def reset(self) -> None:
        """Reset all action controllers."""
        self.mouse.reset()
        self.swipe_detector.reset()

    def process_frame(
        self,
        hand: Optional[HandData],
        classification: ClassificationResult,
        state_snapshot: StateSnapshot,
        timestamp: Optional[float] = None,
    ) -> DispatchResult:
        """Process tracking data for a single frame and dispatch appropriate actions.

        Args:
            hand: Dominant hand data if present.
            classification: Output from static classifier.
            state_snapshot: Current FSM state snapshot.
            timestamp: Optional explicit timestamp.

        Returns:
            DispatchResult describing action performed.
        """
        now = timestamp if timestamp is not None else time.time()

        # If system is not in ACTIVE state, suppress all user-interaction actions
        if not state_snapshot.is_active or hand is None:
            self.mouse.reset()
            return DispatchResult(action_type="IDLE", detail="Control inactive or no hand detected.")

        # 1. Evaluate Dynamic Swipes (takes precedence during fast open-palm motion)
        swipe_ev = self.swipe_detector.update(
            palm_center_norm=hand.palm_center_norm,
            current_gesture=classification.gesture,
            timestamp=now,
        )

        if swipe_ev is not None:
            return self._dispatch_swipe(swipe_ev)

        # 2. Evaluate Static Gestures
        gesture = state_snapshot.stable_gesture
        lms = hand.landmarks

        # A. Point: Move cursor
        if gesture == StaticGesture.POINT:
            # Index fingertip position
            idx_x = lms[INDEX_TIP].x
            idx_y = lms[INDEX_TIP].y
            cx, cy = self.mouse.process_point(idx_x, idx_y)
            # Ensure pinch is released
            self.mouse.handle_pinch(is_pinching=False, timestamp=now)
            return DispatchResult(action_type="MOUSE_MOVE", detail=f"Moving cursor to ({cx}, {cy})", cursor_pos=(cx, cy))

        # B. Pinch: Click or Drag
        elif gesture == StaticGesture.PINCH:
            idx_x = lms[INDEX_TIP].x
            idx_y = lms[INDEX_TIP].y
            self.mouse.handle_pinch(is_pinching=True, current_pos=(idx_x, idx_y), timestamp=now)
            cur = self.mouse.smoother.current_value
            c_pos = (int(cur[0]), int(cur[1])) if cur else None
            return DispatchResult(action_type="MOUSE_PINCH", detail="Pinch active (click/drag)", cursor_pos=c_pos)

        # C. Two Fingers Up: Scroll
        elif gesture == StaticGesture.TWO_FINGERS_UP:
            # Palm or index MCP vertical position
            norm_y = hand.palm_center_norm[1]
            clicks = self.mouse.process_scroll(norm_y)
            return DispatchResult(action_type="MOUSE_SCROLL", detail=f"Scrolling: {clicks} clicks")

        # D. Three Fingers Up: Continuous Volume Slider
        elif gesture == StaticGesture.THREE_FINGERS_UP:
            # Map palm Y position inside active region:
            # Top of active box (min_y) = 1.0 (100% volume)
            # Bottom of active box (max_y) = 0.0 (0% volume)
            min_y, max_y = self.config.active_region.min_y, self.config.active_region.max_y
            raw_y = hand.palm_center_norm[1]
            clamped_y = clamp(raw_y, min_y, max_y)
            # Invert: higher on screen (lower Y) is louder
            vol_target = 1.0 - ((clamped_y - min_y) / max(1e-5, max_y - min_y))
            self.volume.set_volume(vol_target)
            return DispatchResult(
                action_type="VOLUME_SLIDER",
                detail=f"Continuous volume: {vol_target * 100:.0f}%",
                volume_level=vol_target,
            )

        # E. Other gestures (Open Palm, Fist, Unknown): Release active mouse buttons
        else:
            self.mouse.handle_pinch(is_pinching=False, timestamp=now)
            return DispatchResult(action_type="HOVER", detail=f"Passive gesture: {gesture.value}")

    def _dispatch_swipe(self, swipe_ev: SwipeEvent) -> DispatchResult:
        """Handle directional swipe actions."""
        # Horizontal Swipes -> Arrow keys
        if swipe_ev.direction == SwipeDirection.SWIPE_LEFT:
            self.keyboard.trigger_left()
            return DispatchResult(action_type="KEYBOARD", detail="Swipe Left -> Left Arrow")

        elif swipe_ev.direction == SwipeDirection.SWIPE_RIGHT:
            self.keyboard.trigger_right()
            return DispatchResult(action_type="KEYBOARD", detail="Swipe Right -> Right Arrow")

        # Vertical Swipes -> Volume adjustment (step 5%)
        elif swipe_ev.direction == SwipeDirection.SWIPE_UP:
            new_vol = self.volume.step_volume(self.config.gestures.volume_step)
            return DispatchResult(
                action_type="VOLUME_STEP",
                detail=f"Swipe Up -> Volume +5% ({new_vol * 100:.0f}%)",
                volume_level=new_vol,
            )

        elif swipe_ev.direction == SwipeDirection.SWIPE_DOWN:
            new_vol = self.volume.step_volume(-self.config.gestures.volume_step)
            return DispatchResult(
                action_type="VOLUME_STEP",
                detail=f"Swipe Down -> Volume -5% ({new_vol * 100:.0f}%)",
                volume_level=new_vol,
            )

        return DispatchResult(action_type="SWIPE_UNKNOWN", detail="Unknown swipe direction")

