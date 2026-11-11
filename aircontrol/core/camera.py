"""Threaded camera capture engine for AirControl.

Decouples image acquisition from processing to guarantee >= 25 FPS.
Drops stale frames, calculates real-time FPS, monitors luminance for low-light warnings,
and gracefully recovers when the camera is disconnected or in use by another application.
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from aircontrol.utils.logger import get_logger
from aircontrol.core.config import CameraConfig

logger = get_logger("camera")


@dataclass
class CameraFrame:
    """Encapsulates a captured webcam frame and its associated telemetry."""
    frame: np.ndarray
    timestamp: float
    fps: float
    is_low_light: bool
    mean_luminance: float
    is_default_camera: bool = False


class CameraThread:
    """Dedicated background thread for non-blocking OpenCV video capture."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Atomic buffer holding latest frame (stale frames overwritten immediately)
        self._latest_frame: Optional[CameraFrame] = None
        self._is_connected = False
        self._error_message: Optional[str] = None
        self._is_default_camera = False
        self._force_default_camera = False

        # Telemetry
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0

    @property
    def is_connected(self) -> bool:
        """Whether the camera is currently opened and acquiring frames."""
        with self._lock:
            return self._is_connected

    @property
    def is_default_camera(self) -> bool:
        """Whether the camera stream is currently using the default virtual fallback."""
        with self._lock:
            return self._is_default_camera

    def switch_to_default_camera(self) -> None:
        """Switch to default virtual camera stream."""
        with self._lock:
            self._force_default_camera = True
            self._is_default_camera = True
        self._release_camera()
        logger.info("Switched to Default Virtual Camera.")

    def retry_hardware_camera(self) -> bool:
        """Attempt to connect to physical webcam hardware."""
        with self._lock:
            self._force_default_camera = False
        success = self._open_camera()
        if success:
            with self._lock:
                self._is_default_camera = False
            logger.info("Successfully connected to physical hardware camera.")
        return success

    @property
    def error_message(self) -> Optional[str]:
        """Description of the last camera acquisition error, if any."""
        with self._lock:
            return self._error_message

    def start(self) -> bool:
        """Start background capture thread.

        Returns:
            True if started successfully, False if already running or failed to initialize.
        """
        if self._running:
            logger.warning("CameraThread is already running.")
            return True

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="AirControlCameraThread", daemon=True)
        self._thread.start()
        logger.info("Camera capture thread started for camera index %d.", self.config.camera_index)
        return True

    def stop(self) -> None:
        """Signal capture thread to terminate and release hardware resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
            self._thread = None

        self._release_camera()
        logger.info("Camera capture thread stopped and camera released.")

    def _open_camera(self) -> bool:
        """Attempt to open hardware video capture across supported backends."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(self.config.camera_index, backend)
                if cap.isOpened():
                    self._cap = cap
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                    self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

                    with self._lock:
                        self._is_connected = True
                        self._error_message = None
                    logger.info(
                        "Camera index %d successfully opened with backend %d (%dx%d @ %d FPS).",
                        self.config.camera_index, backend, self.config.width, self.config.height, self.config.target_fps
                    )
                    return True
                else:
                    cap.release()
            except Exception as exc:
                logger.debug("Backend %d failed: %s", backend, exc)

        with self._lock:
            self._is_connected = False
            self._error_message = (
                f"Camera {self.config.camera_index} is unavailable or privacy-blocked."
            )
        return False

    def _release_camera(self) -> None:
        """Safely release the VideoCapture resource."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("Error releasing camera: %s", exc)
            finally:
                self._cap = None
        with self._lock:
            self._is_connected = False

    def _capture_loop(self) -> None:
        """Worker loop executing continuous frame grab and buffer updates."""
        if not self._open_camera():
            # If camera opening failed, wait and retry periodically
            pass

        reconnect_interval = 2.0  # seconds between reconnect attempts
        last_reconnect_time = time.time()

        while self._running:
            if self._force_default_camera or (self.config.use_default_fallback and (self._cap is None or not self._cap.isOpened())):
                with self._lock:
                    self._is_default_camera = True
                    self._is_connected = True
                    self._error_message = None

                now = time.time()
                self._frame_count += 1
                elapsed = now - self._fps_start_time
                if elapsed >= 1.0:
                    self._current_fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_start_time = now

                # Generate clean synthetic camera feed
                virtual_frame = np.full((self.config.height, self.config.width, 3), 18, dtype=np.uint8)
                virtual_frame[::40, :, :] = 32
                virtual_frame[:, ::40, :] = 32

                cam_frame = CameraFrame(
                    frame=virtual_frame,
                    timestamp=now,
                    fps=self._current_fps if self._current_fps > 0 else 30.0,
                    is_low_light=False,
                    mean_luminance=35.0,
                    is_default_camera=True,
                )

                with self._lock:
                    self._latest_frame = cam_frame

                time.sleep(1.0 / self.config.target_fps)

                # Periodic background retry for physical camera if not forced
                if not self._force_default_camera and (time.time() - last_reconnect_time >= reconnect_interval):
                    last_reconnect_time = time.time()
                    if self._open_camera():
                        with self._lock:
                            self._is_default_camera = False
                continue

            if self._cap is None or not self._cap.isOpened():
                if time.time() - last_reconnect_time >= reconnect_interval:
                    last_reconnect_time = time.time()
                    logger.debug("Attempting to reconnect camera...")
                    self._open_camera()
                time.sleep(0.05)
                continue

            success, raw_frame = self._cap.read()
            if not success or raw_frame is None:
                with self._lock:
                    self._is_connected = False
                    self._error_message = "Failed to read frame from camera."
                logger.warning("Dropped frame or read failure. Releasing camera for reconnect...")
                self._release_camera()
                time.sleep(0.1)
                continue

            now = time.time()
            self._frame_count += 1
            elapsed = now - self._fps_start_time
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = now

            # Calculate mean luminance for low-light detection (fast downsampled mean)
            small_gray = cv2.cvtColor(raw_frame[::4, ::4], cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(small_gray))
            is_low_light = mean_lum < self.config.low_light_threshold

            # Atomically update latest frame (stale frame dropped automatically)
            camera_frame = CameraFrame(
                frame=raw_frame,
                timestamp=now,
                fps=self._current_fps,
                is_low_light=is_low_light,
                mean_luminance=mean_lum,
            )

            with self._lock:
                self._latest_frame = camera_frame
                self._is_connected = True
                self._error_message = None

        self._release_camera()

    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Retrieve the latest captured frame atomically without blocking.

        Returns:
            Latest CameraFrame, or None if no frame is available yet.
        """
        with self._lock:
            return self._latest_frame

