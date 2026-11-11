"""Finite State Machine (FSM) and gesture stabilization engine for AirControl.

Provides:
- FSM States: IDLE (standby) <-> ARMED (fist held) <-> ACTIVE (full control)
- 1.0s Fist Hold Timer with progress tracking for on-screen HUD animation
- N-consecutive frame debouncing to eliminate gesture noise and transition flickers
"""

from __future__ import annotations
import collections
import time
from enum import Enum
from dataclasses import dataclass
from typing import Deque, Optional, Tuple, Callable, List

from aircontrol.core.classifier import StaticGesture
from aircontrol.core.config import GestureThresholds
from aircontrol.utils.logger import get_logger

logger = get_logger("state")


class ControlState(str, Enum):
    """System control operational state."""
    IDLE = "IDLE"      # System inactive, ignoring cursor and shortcut actions
    ARMED = "ARMED"    # Fist is held, accumulating time towards state toggle
    ACTIVE = "ACTIVE"  # Full gesture control active


@dataclass
class StateSnapshot:
    """Telemetry snapshot of current state machine condition."""
    control_state: ControlState
    stable_gesture: StaticGesture
    raw_gesture: StaticGesture
    fist_hold_progress: float  # [0.0, 1.0]
    is_active: bool


class ControlStateMachine:
    """Manages application state transitions and gesture frame stabilization."""

    def __init__(self, thresholds: GestureThresholds) -> None:
        self.thresholds = thresholds
        self._control_state = ControlState.IDLE
        self._target_state_on_fist_complete = ControlState.ACTIVE

        # Fist hold timer
        self._fist_start_time: Optional[float] = None
        self._hold_progress: float = 0.0

        # Frame stabilizer: N consecutive frames required to confirm static gesture
        self._history_len = max(1, thresholds.consecutive_frames_required)
        self._gesture_history: Deque[StaticGesture] = collections.deque(maxlen=self._history_len)
        self._stable_gesture = StaticGesture.NONE

        # State transition callbacks
        self._on_state_change_callbacks: List[Callable[[ControlState], None]] = []

    @property
    def current_state(self) -> ControlState:
        return self._control_state

    @property
    def stable_gesture(self) -> StaticGesture:
        return self._stable_gesture

    @property
    def is_active(self) -> bool:
        return self._control_state == ControlState.ACTIVE

    def register_state_change_callback(self, cb: Callable[[ControlState], None]) -> None:
        """Register listener for state changes (IDLE <-> ACTIVE)."""
        self._on_state_change_callbacks.append(cb)

    def _set_state(self, new_state: ControlState) -> None:
        """Transition to a new control state and notify listeners."""
        if self._control_state != new_state:
            old_state = self._control_state
            self._control_state = new_state
            logger.info("ControlState changed: %s -> %s", old_state.value, new_state.value)
            for cb in self._on_state_change_callbacks:
                try:
                    cb(new_state)
                except Exception as exc:
                    logger.exception("Error in state change callback: %s", exc)

    def update(
        self,
        raw_gesture: StaticGesture,
        timestamp: Optional[float] = None,
    ) -> StateSnapshot:
        """Process incoming raw gesture and advance FSM and timers.

        Args:
            raw_gesture: Raw StaticGesture from current frame classifier.
            timestamp: Optional explicit timestamp for test determinism.

        Returns:
            StateSnapshot containing updated state telemetry.
        """
        now = timestamp if timestamp is not None else time.time()

        # 1. Update N-frame history and stable gesture
        self._gesture_history.append(raw_gesture)
        if len(self._gesture_history) == self._history_len:
            first = self._gesture_history[0]
            if all(g == first for g in self._gesture_history):
                self._stable_gesture = first

        # 2. Fist Hold State Logic (toggles IDLE <-> ACTIVE on 1.0s hold)
        fist_duration_target = self.thresholds.fist_hold_duration_s

        if raw_gesture == StaticGesture.FIST:
            if self._fist_start_time is None:
                self._fist_start_time = now
                # Decide which state we are aiming for when hold completes
                if self._control_state == ControlState.IDLE:
                    self._target_state_on_fist_complete = ControlState.ACTIVE
                else:
                    self._target_state_on_fist_complete = ControlState.IDLE

            elapsed = now - self._fist_start_time
            self._hold_progress = min(1.0, elapsed / fist_duration_target)

            if elapsed >= fist_duration_target:
                # Successfully held fist for 1s: toggle state!
                self._set_state(self._target_state_on_fist_complete)
                self._fist_start_time = None
                self._hold_progress = 0.0
            else:
                if self._control_state != ControlState.ARMED:
                    self._control_state = ControlState.ARMED
        else:
            # Fist released before 1.0s: cancel arming
            if self._control_state == ControlState.ARMED:
                # Revert back to original base state
                orig = ControlState.ACTIVE if self._target_state_on_fist_complete == ControlState.IDLE else ControlState.IDLE
                self._set_state(orig)

            self._fist_start_time = None
            self._hold_progress = 0.0

        return StateSnapshot(
            control_state=self._control_state,
            stable_gesture=self._stable_gesture,
            raw_gesture=raw_gesture,
            fist_hold_progress=self._hold_progress,
            is_active=self.is_active,
        )

    def force_toggle(self) -> ControlState:
        """Programmatically toggle state between IDLE and ACTIVE (e.g. from UI button)."""
        new_state = ControlState.IDLE if self.is_active else ControlState.ACTIVE
        self._set_state(new_state)
        self._fist_start_time = None
        self._hold_progress = 0.0
        return self._control_state

