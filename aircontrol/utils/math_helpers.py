"""Mathematical utilities and signal filtering helpers for AirControl.

Provides:
- Exponential Moving Average (EMA) filters for jitter-free cursor tracking
- Active region box coordinate mapping (normalized camera space -> screen space)
- Deadzone filtering for micro-tremor suppression
- Euclidean distance, vector angles, and displacement calculations
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional


def euclidean_distance_2d(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def euclidean_distance_3d(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float]
) -> float:
    """Calculate 3D Euclidean distance between two points."""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamp a value within [min_val, max_val]."""
    return max(min_val, min(max_val, val))


def map_active_region_to_screen(
    norm_x: float,
    norm_y: float,
    active_box: Tuple[float, float, float, float],
    screen_width: int,
    screen_height: int,
    flip_horizontal: bool = True,
) -> Tuple[int, int]:
    """Map normalized coordinates (0.0 - 1.0) inside an active region box to screen coordinates."""
    min_x, min_y, max_x, max_y = active_box
    box_w = max(1e-5, max_x - min_x)
    box_h = max(1e-5, max_y - min_y)

    effective_x = (1.0 - norm_x) if flip_horizontal else norm_x
    clamped_x = clamp(effective_x, min_x, max_x)
    clamped_y = clamp(norm_y, min_y, max_y)

    box_norm_x = (clamped_x - min_x) / box_w
    box_norm_y = (clamped_y - min_y) / box_h

    screen_x = int(box_norm_x * (screen_width - 1))
    screen_y = int(box_norm_y * (screen_height - 1))

    return screen_x, screen_y


class ExponentialSmoothing2D:
    """Exponential Moving Average (EMA) filter with deadzone suppression."""

    def __init__(self, alpha: float = 0.35, deadzone: float = 3.0) -> None:
        self.alpha = clamp(alpha, 0.01, 1.0)
        self.deadzone = max(0.0, deadzone)
        self._current_x: Optional[float] = None
        self._current_y: Optional[float] = None

    def reset(self) -> None:
        self._current_x = None
        self._current_y = None

    def update(self, target_x: float, target_y: float) -> Tuple[float, float]:
        if self._current_x is None or self._current_y is None:
            self._current_x = float(target_x)
            self._current_y = float(target_y)
            return self._current_x, self._current_y

        dist = math.hypot(target_x - self._current_x, target_y - self._current_y)
        if dist < self.deadzone:
            return self._current_x, self._current_y

        self._current_x = self.alpha * target_x + (1.0 - self.alpha) * self._current_x
        self._current_y = self.alpha * target_y + (1.0 - self.alpha) * self._current_y

        return self._current_x, self._current_y

    @property
    def current_value(self) -> Optional[Tuple[float, float]]:
        if self._current_x is None or self._current_y is None:
            return None
        return self._current_x, self._current_y
