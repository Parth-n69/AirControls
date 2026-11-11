"""Mouse control subsystem for AirControl.

Provides:
- Exponential Moving Average (EMA) cursor smoothing
- Active region box mapping (maps camera zone to full desktop resolution)
- Deadzone suppression to eliminate physiological micro-tremors
- Left-click on quick pinch
- Click-and-drag on pinch-and-hold
- Two-finger vertical and horizontal scrolling
- Decoupled backend architecture for headless unit testing
"""

from __future__ import annotations
import time
from typing import Tuple, Optional, Protocol
import pyautogui

from aircontrol.core.config import MouseConfig, ActiveRegionConfig
from aircontrol.utils.math_helpers import (
    map_active_region_to_screen,
    ExponentialSmoothing2D,
)
from aircontrol.utils.logger import get_logger

logger = get_logger("mouse")


class MouseBackend(Protocol):
    """Protocol defining the interface for OS mouse dispatchers."""
    def get_screen_size(self) -> Tuple[int, int]: ...
    def move_to(self, x: int, y: int) -> None: ...
    def mouse_down(self, button: str = "left") -> None: ...
    def mouse_up(self, button: str = "left") -> None: ...
    def click(self, button: str = "left") -> None: ...
    def scroll(self, clicks: int) -> None: ...


class PyAutoGUIMouseBackend:
    """Production mouse backend powered by PyAutoGUI."""

    def __init__(self) -> None:
        # Zero PyAutoGUI per-action pause for real-time response
        pyautogui.PAUSE = 0.0
        pyautogui.FAILSAFE = False

    def get_screen_size(self) -> Tuple[int, int]:
        w, h = pyautogui.size()
        return int(w), int(h)

    def move_to(self, x: int, y: int) -> None:
        pyautogui.moveTo(x, y)

    def mouse_down(self, button: str = "left") -> None:
        pyautogui.mouseDown(button=button)

    def mouse_up(self, button: str = "left") -> None:
        pyautogui.mouseUp(button=button)

    def click(self, button: str = "left") -> None:
        pyautogui.click(button=button)

    def scroll(self, clicks: int) -> None:
        pyautogui.scroll(clicks)


class MockMouseBackend:
    """Mock mouse backend for deterministic unit tests and headless environments."""

    def __init__(self, screen_size: Tuple[int, int] = (1920, 1080)) -> None:
        self.screen_width, self.screen_height = screen_size
        self.cursor_pos: Tuple[int, int] = (self.screen_width // 2, self.screen_height // 2)
        self.is_down = False
        self.clicks_count = 0
        self.scroll_total = 0

    def get_screen_size(self) -> Tuple[int, int]:
        return self.screen_width, self.screen_height

    def move_to(self, x: int, y: int) -> None:
        self.cursor_pos = (x, y)

    def mouse_down(self, button: str = "left") -> None:
        self.is_down = True

    def mouse_up(self, button: str = "left") -> None:
        self.is_down = False

    def click(self, button: str = "left") -> None:
        self.clicks_count += 1

    def scroll(self, clicks: int) -> None:
        self.scroll_total += clicks


class MouseController:
    """High-level mouse interaction controller with smoothing, drag, and scroll."""

    def __init__(
        self,
        mouse_config: MouseConfig,
        active_region: ActiveRegionConfig,
        backend: Optional[MouseBackend] = None,
        flip_horizontal: bool = True,
    ) -> None:
        self.config = mouse_config
        self.active_region = active_region
        self.flip_horizontal = flip_horizontal
        self.backend = backend or PyAutoGUIMouseBackend()

        try:
            self.screen_width, self.screen_height = self.backend.get_screen_size()
        except Exception:
            self.screen_width, self.screen_height = 1920, 1080

        self.smoother = ExponentialSmoothing2D(
            alpha=self.config.smoothing_alpha,
            deadzone=self.config.deadzone_pixels,
        )

        # Pinch & Drag State
        self._is_pinching = False
        self._pinch_start_time: Optional[float] = None
        self._is_dragging = False

        # Scroll State
        self._prev_scroll_y: Optional[float] = None

    def reset(self) -> None:
        """Reset internal filter and button states."""
        if self._is_dragging or self._is_pinching:
            try:
                self.backend.mouse_up(button="left")
            except Exception:
                pass
        self._is_pinching = False
        self._is_dragging = False
        self._pinch_start_time = None
        self._prev_scroll_y = None
        self.smoother.reset()

    def update_config(self, mouse_config: MouseConfig, active_region: ActiveRegionConfig) -> None:
        """Update runtime mouse sensitivity and active region settings."""
        self.config = mouse_config
        self.active_region = active_region
        self.smoother.alpha = mouse_config.smoothing_alpha
        self.smoother.deadzone = mouse_config.deadzone_pixels

    def process_point(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Move cursor based on index fingertip position with smoothing and active zone mapping.

        Returns:
            Smoothed (screen_x, screen_y) pixels.
        """
        raw_sx, raw_sy = map_active_region_to_screen(
            norm_x=norm_x,
            norm_y=norm_y,
            active_box=self.active_region.as_tuple(),
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            flip_horizontal=self.flip_horizontal,
        )

        smooth_sx, smooth_sy = self.smoother.update(raw_sx, raw_sy)
        target_x, target_y = int(smooth_sx), int(smooth_sy)

        self.backend.move_to(target_x, target_y)
        return target_x, target_y

    def handle_pinch(self, is_pinching: bool, current_pos: Optional[Tuple[float, float]] = None, timestamp: Optional[float] = None) -> None:
        """Manage left click and pinch-and-hold drag transitions.

        Args:
            is_pinching: Whether pinch gesture is currently detected.
            current_pos: Optional (norm_x, norm_y) to move cursor while dragging.
            timestamp: Optional explicit time in seconds.
        """
        now = timestamp if timestamp is not None else time.time()

        if is_pinching:
            if not self._is_pinching:
                # Pinch started
                self._is_pinching = True
                self._pinch_start_time = now
                self._is_dragging = False
                logger.debug("Pinch initiated at %.3f", now)
            else:
                # Continuing pinch: check if held long enough to initiate drag
                hold_duration = now - (self._pinch_start_time or now)
                if not self._is_dragging and hold_duration >= self.config.drag_hold_delay_s:
                    self._is_dragging = True
                    self.backend.mouse_down(button="left")
                    logger.info("Pinch-and-hold triggered: DRAG STARTED.")

                # If dragging and a position is supplied, continue moving cursor
                if self._is_dragging and current_pos is not None:
                    self.process_point(current_pos[0], current_pos[1])

        else:
            if self._is_pinching:
                # Pinch released
                hold_duration = now - (self._pinch_start_time or now)
                if self._is_dragging:
                    # Release drag
                    self.backend.mouse_up(button="left")
                    logger.info("Drag released.")
                else:
                    # Short pinch = single click
                    self.backend.click(button="left")
                    logger.info("Quick pinch triggered: CLICK.")

                self._is_pinching = False
                self._is_dragging = False
                self._pinch_start_time = None

    def process_scroll(self, norm_y: float) -> int:
        """Process vertical scroll based on two-finger movement delta.

        Args:
            norm_y: Normalized vertical position of hand.

        Returns:
            Number of scroll clicks dispatched.
        """
        if self._prev_scroll_y is None:
            self._prev_scroll_y = norm_y
            return 0

        dy = norm_y - self._prev_scroll_y
        self._prev_scroll_y = norm_y

        if abs(dy) < self.config.scroll_deadzone:
            return 0

        # Invert direction: moving hand UP (negative norm_y delta) scrolls UP (+clicks)
        scroll_clicks = int(-dy * 100 * self.config.scroll_factor)
        if scroll_clicks != 0:
            self.backend.scroll(scroll_clicks)
            logger.debug("Dispatched scroll clicks: %d", scroll_clicks)

        return scroll_clicks

