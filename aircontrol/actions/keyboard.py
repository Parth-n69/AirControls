"""Keyboard action dispatcher for AirControl.

Dispatches keyboard commands for directional swipe navigation:
- Open palm swipe LEFT  -> Left arrow / previous
- Open palm swipe RIGHT -> Right arrow / next
- Custom configurable key mappings
"""

from __future__ import annotations
import time
from typing import Optional, Protocol, List
import pyautogui

from aircontrol.utils.logger import get_logger

logger = get_logger("keyboard")


class KeyboardBackend(Protocol):
    """Protocol for keyboard action dispatchers."""
    def press_key(self, key: str) -> None: ...


class PyAutoGUIKeyboardBackend:
    """Production keyboard dispatcher using PyAutoGUI."""

    def press_key(self, key: str) -> None:
        pyautogui.press(key)


class MockKeyboardBackend:
    """Mock keyboard dispatcher for unit testing."""

    def __init__(self) -> None:
        self.pressed_keys: List[str] = []

    def press_key(self, key: str) -> None:
        self.pressed_keys.append(key)


class KeyboardController:
    """Dispatches navigational and shortcut keyboard inputs with debouncing."""

    def __init__(self, backend: Optional[KeyboardBackend] = None, min_key_interval_s: float = 0.25) -> None:
        self.backend = backend or PyAutoGUIKeyboardBackend()
        self.min_key_interval_s = min_key_interval_s
        self._last_press_time: float = 0.0

    def trigger_left(self) -> bool:
        """Dispatch Left Arrow key (e.g. for Previous slide or page)."""
        return self._press("left")

    def trigger_right(self) -> bool:
        """Dispatch Right Arrow key (e.g. for Next slide or page)."""
        return self._press("right")

    def trigger_key(self, key_name: str) -> bool:
        """Dispatch an arbitrary key with debouncing."""
        return self._press(key_name)

    def _press(self, key: str) -> bool:
        now = time.time()
        if now - self._last_press_time < self.min_key_interval_s:
            logger.debug("Key '%s' suppressed by debounce window.", key)
            return False

        self._last_press_time = now
        try:
            self.backend.press_key(key)
            logger.info("Dispatched key press: '%s'", key)
            return True
        except Exception as exc:
            logger.error("Failed to press key '%s': %s", key, exc)
            return False

