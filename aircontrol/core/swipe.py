"""Dynamic swipe gesture detection with trajectory, velocity, and direction lock.

Tracks palm centroid movement over a sliding time window.
Requires:
1. Significant displacement above distance threshold
2. Peak velocity above minimum threshold
3. Direction dominance lock (e.g. horizontal displacement > 1.8 * vertical displacement)
4. Cooldown timer guaranteeing exactly one action per physical swipe
"""

from __future__ import annotations
import collections
import time
from enum import Enum
from dataclasses import dataclass
from typing import Deque, Tuple, Optional

from aircontrol.core.config import GestureThresholds
from aircontrol.core.classifier import StaticGesture
from aircontrol.utils.logger import get_logger

logger = get_logger("swipe")


class SwipeDirection(str, Enum):
    """Directions for dynamic swipes."""
    NONE = "NONE"
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    SWIPE_UP = "SWIPE_UP"
    SWIPE_DOWN = "SWIPE_DOWN"


@dataclass
class TrajectoryPoint:
    """Historical timestamped position of palm center."""
    x: float
    y: float
    timestamp: float


@dataclass
class SwipeEvent:
    """Event triggered upon successful swipe recognition."""
    direction: SwipeDirection
    displacement_x: float
    displacement_y: float
    velocity_x: float
    velocity_y: float
    timestamp: float


class SwipeDetector:
    """Detects directional swipes with direction locking and debounce."""

    def __init__(self, thresholds: GestureThresholds, flip_horizontal: bool = True) -> None:
        self.thresholds = thresholds
        self.flip_horizontal = flip_horizontal
        self._history: Deque[TrajectoryPoint] = collections.deque()
        self._last_trigger_time: float = 0.0

    def reset(self) -> None:
        """Clear motion history buffer."""
        self._history.clear()

    def update(
        self,
        palm_center_norm: Tuple[float, float],
        current_gesture: StaticGesture,
        timestamp: Optional[float] = None,
    ) -> Optional[SwipeEvent]:
        """Feed current palm position and evaluate swipe conditions.

        Args:
            palm_center_norm: (x, y) normalized coordinates in [0.0, 1.0].
            current_gesture: Current static hand pose (must be OPEN_PALM to swipe).
            timestamp: Optional explicit timestamp (seconds), defaults to time.time().

        Returns:
            SwipeEvent if a valid swipe gesture is recognized, None otherwise.
        """
        now = timestamp if timestamp is not None else time.time()

        # In cooldown period: suppress any new triggers
        if now - self._last_trigger_time < self.thresholds.swipe_cooldown_s:
            return None

        # Swipes only engage when the hand is in OPEN_PALM posture
        if current_gesture != StaticGesture.OPEN_PALM:
            self._history.clear()
            return None

        # Effective X orientation (if flipped/mirrored, user moving hand to their right increases X)
        raw_x, raw_y = palm_center_norm
        eff_x = (1.0 - raw_x) if self.flip_horizontal else raw_x
        eff_y = raw_y

        self._history.append(TrajectoryPoint(x=eff_x, y=eff_y, timestamp=now))

        # Evict points outside the sliding time window
        window_s = self.thresholds.swipe_window_s
        while self._history and (now - self._history[0].timestamp > window_s):
            self._history.popleft()

        # Need at least 3 points across at least 80ms of trajectory
        if len(self._history) < 3:
            return None

        start_pt = self._history[0]
        end_pt = self._history[-1]
        dt = end_pt.timestamp - start_pt.timestamp

        if dt < 0.08:  # Minimum duration to distinguish from instant noise
            return None

        dx = end_pt.x - start_pt.x
        dy = end_pt.y - start_pt.y  # In screen coords, positive Y is DOWN, negative Y is UP

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        vx = dx / dt
        vy = dy / dt
        speed = (abs_dx**2 + abs_dy**2) ** 0.5 / dt

        # Check minimum displacement and velocity criteria
        min_dist = self.thresholds.swipe_min_distance
        min_vel = self.thresholds.swipe_min_velocity

        if speed < min_vel:
            return None

        lock_ratio = self.thresholds.swipe_direction_ratio
        eps = 1e-5

        detected_direction = SwipeDirection.NONE

        # Horizontal swipe check (dominant X)
        if abs_dx >= min_dist and (abs_dx / (abs_dy + eps)) >= lock_ratio:
            if dx > 0:
                detected_direction = SwipeDirection.SWIPE_RIGHT
            else:
                detected_direction = SwipeDirection.SWIPE_LEFT

        # Vertical swipe check (dominant Y)
        elif abs_dy >= min_dist and (abs_dy / (abs_dx + eps)) >= lock_ratio:
            if dy < 0:
                detected_direction = SwipeDirection.SWIPE_UP
            else:
                detected_direction = SwipeDirection.SWIPE_DOWN

        if detected_direction != SwipeDirection.NONE:
            self._last_trigger_time = now
            self._history.clear()
            logger.info(
                "Swipe detected: %s (dx=%.3f, dy=%.3f, speed=%.2f/s)",
                detected_direction.value, dx, dy, speed
            )
            return SwipeEvent(
                direction=detected_direction,
                displacement_x=dx,
                displacement_y=dy,
                velocity_x=vx,
                velocity_y=vy,
                timestamp=now,
            )

        return None

