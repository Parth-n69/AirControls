"""Volume controller subsystem with clean platform abstraction.

Provides:
- Abstract Base Class VolumeController (enabling seamless macOS/Linux extension)
- Concrete WindowsVolumeController powered by pycaw and Windows Core Audio APIs
- MockVolumeController for headless unit testing and non-Windows fallback
- Step volume adjustment (+/- 5% per swipe UP/DOWN)
- Continuous vertical slider tracking (3 fingers up)
"""

from __future__ import annotations
import platform
from abc import ABC, abstractmethod
from typing import Optional

from aircontrol.utils.logger import get_logger
from aircontrol.utils.math_helpers import clamp

logger = get_logger("volume")


class VolumeController(ABC):
    """Abstract interface for operating system audio volume."""

    @abstractmethod
    def get_volume(self) -> float:
        """Return master volume level in [0.0, 1.0]."""
        pass

    @abstractmethod
    def set_volume(self, volume: float) -> None:
        """Set master volume level in [0.0, 1.0]."""
        pass

    @abstractmethod
    def step_volume(self, delta: float) -> float:
        """Adjust master volume by delta (+/- 0.05 for 5% step).

        Returns:
            Updated volume level in [0.0, 1.0].
        """
        pass

    @abstractmethod
    def is_muted(self) -> bool:
        """Return True if audio output is muted."""
        pass

    @abstractmethod
    def set_mute(self, mute: bool) -> None:
        """Set mute state."""
        pass


class WindowsVolumeController(VolumeController):
    """Windows audio controller via pycaw / IAudioEndpointVolume."""

    def __init__(self) -> None:
        self._endpoint = None
        self._initialize_pycaw()

    def _initialize_pycaw(self) -> None:
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from comtypes import CLSCTX_ALL
            import ctypes

            # Initialize COM for the calling thread
            try:
                ctypes.windll.ole32.CoInitialize(None)
            except Exception:
                pass

            devices = AudioUtilities.GetSpeakers()
            if devices is None:
                raise RuntimeError("No default audio output speakers found.")

            if hasattr(devices, "EndpointVolume") and devices.EndpointVolume is not None:
                self._endpoint = devices.EndpointVolume
            elif hasattr(devices, "Activate"):
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._endpoint = interface.QueryInterface(IAudioEndpointVolume)
            else:
                raise RuntimeError("Unsupported AudioDevice interface.")
            logger.info("WindowsVolumeController initialized successfully via pycaw.")
        except Exception as exc:
            logger.warning("Failed to initialize Windows pycaw audio endpoint: %s. Using mock fallback.", exc)
            self._endpoint = None

    def get_volume(self) -> float:
        if self._endpoint is None:
            return 0.50
        try:
            return float(self._endpoint.GetMasterVolumeLevelScalar())
        except Exception as exc:
            logger.error("Error reading volume: %s", exc)
            return 0.50

    def set_volume(self, volume: float) -> None:
        val = clamp(volume, 0.0, 1.0)
        if self._endpoint is None:
            return
        try:
            self._endpoint.SetMasterVolumeLevelScalar(val, None)
        except Exception as exc:
            logger.error("Error setting volume to %.2f: %s", val, exc)

    def step_volume(self, delta: float) -> float:
        current = self.get_volume()
        target = clamp(current + delta, 0.0, 1.0)
        self.set_volume(target)
        logger.info("Volume adjusted by %.2f: %.0f%% -> %.0f%%", delta, current * 100, target * 100)
        return target

    def is_muted(self) -> bool:
        if self._endpoint is None:
            return False
        try:
            return bool(self._endpoint.GetMute())
        except Exception as exc:
            logger.error("Error reading mute: %s", exc)
            return False

    def set_mute(self, mute: bool) -> None:
        if self._endpoint is None:
            return
        try:
            self._endpoint.SetMute(1 if mute else 0, None)
            logger.info("Mute set to %s", mute)
        except Exception as exc:
            logger.error("Error setting mute: %s", exc)


class MockVolumeController(VolumeController):
    """In-memory mock volume controller for testing or non-Windows platforms."""

    def __init__(self, initial_volume: float = 0.50) -> None:
        self._volume = clamp(initial_volume, 0.0, 1.0)
        self._muted = False

    def get_volume(self) -> float:
        return self._volume

    def set_volume(self, volume: float) -> None:
        self._volume = clamp(volume, 0.0, 1.0)

    def step_volume(self, delta: float) -> float:
        self._volume = clamp(self._volume + delta, 0.0, 1.0)
        return self._volume

    def is_muted(self) -> bool:
        return self._muted

    def set_mute(self, mute: bool) -> None:
        self._muted = mute


def get_volume_controller() -> VolumeController:
    """Factory creating the appropriate VolumeController for the current operating system."""
    if platform.system() == "Windows":
        try:
            controller = WindowsVolumeController()
            if controller._endpoint is not None:
                return controller
        except Exception as exc:
            logger.warning("WindowsVolumeController fallback due to: %s", exc)

    logger.info("Using MockVolumeController.")
    return MockVolumeController()

