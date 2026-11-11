"""Configuration management for AirControl.

Handles loading, validating, and persisting user preferences and gesture parameters
from config.json with fallback defaults.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any
from aircontrol.utils.logger import get_logger

logger = get_logger("config")

DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class CameraConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    inference_width: int = 480
    inference_height: int = 270
    target_fps: int = 30
    flip_horizontal: bool = True
    low_light_threshold: float = 30.0  # Mean grayscale luminance threshold
    use_default_fallback: bool = True  # Automatically fall back to default virtual camera when no webcam is detected


@dataclass
class ActiveRegionConfig:
    # Normalized coordinates [min_x, min_y, max_x, max_y]
    min_x: float = 0.20
    min_y: float = 0.20
    max_x: float = 0.80
    max_y: float = 0.80

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)


@dataclass
class MouseConfig:
    smoothing_alpha: float = 0.35      # EMA weight for new frame
    deadzone_pixels: float = 3.0       # Ignore jitter below this radius
    speed_multiplier: float = 1.0      # Cursor sensitivity scaling
    drag_hold_delay_s: float = 0.25    # Hold pinch this long to initiate drag
    scroll_factor: float = 1.5         # Scroll sensitivity multiplier
    scroll_deadzone: float = 0.015     # Vertical movement threshold to scroll


@dataclass
class GestureThresholds:
    fist_hold_duration_s: float = 1.0  # Hold fist to toggle control ON/OFF
    pinch_distance_ratio: float = 0.055 # Normalized thumb-index tip distance
    pinch_release_ratio: float = 0.075  # Hysteresis release distance
    finger_straight_threshold: float = 0.12 # Extension threshold
    fist_curl_threshold: float = 0.08   # Finger curl threshold
    swipe_min_distance: float = 0.15   # Normalized displacement in active window
    swipe_min_velocity: float = 0.45   # Normalized distance per second
    swipe_window_s: float = 0.35       # Rolling analysis time window
    swipe_direction_ratio: float = 1.8 # Dominance ratio (dx/dy or dy/dx)
    swipe_cooldown_s: float = 0.50     # Cooldown period after a recognized swipe
    volume_step: float = 0.05          # 5% step per swipe UP/DOWN
    consecutive_frames_required: int = 3 # Consecutive frames to confirm static pose


@dataclass
class UserPreferences:
    dominant_hand: str = "Right"       # "Right" or "Left"
    calibration_scale: float = 1.0     # Hand scale relative to standard calibrated baseline
    start_in_tray: bool = False
    show_landmark_skeleton: bool = True
    show_active_region_box: bool = True
    show_hud_stats: bool = True


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    active_region: ActiveRegionConfig = field(default_factory=ActiveRegionConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    gestures: GestureThresholds = field(default_factory=GestureThresholds)
    preferences: UserPreferences = field(default_factory=UserPreferences)

    @classmethod
    def load(cls, file_path: str = DEFAULT_CONFIG_PATH) -> "AppConfig":
        """Load configuration from JSON file or return defaults if missing/invalid."""
        if not os.path.exists(file_path):
            logger.info("Config file '%s' not found. Creating with default settings.", file_path)
            config = cls()
            config.save(file_path)
            return config

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            camera = CameraConfig(**data.get("camera", {}))
            active_region = ActiveRegionConfig(**data.get("active_region", {}))
            mouse = MouseConfig(**data.get("mouse", {}))
            gestures = GestureThresholds(**data.get("gestures", {}))
            preferences = UserPreferences(**data.get("preferences", {}))

            config = cls(
                camera=camera,
                active_region=active_region,
                mouse=mouse,
                gestures=gestures,
                preferences=preferences,
            )
            logger.info("Configuration successfully loaded from '%s'.", file_path)
            return config
        except Exception as exc:
            logger.error("Failed to parse config file '%s': %s. Reverting to defaults.", file_path, exc)
            return cls()

    def save(self, file_path: str = DEFAULT_CONFIG_PATH) -> bool:
        """Persist current configuration to JSON file."""
        try:
            data = asdict(self)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info("Configuration persisted to '%s'.", file_path)
            return True
        except Exception as exc:
            logger.error("Failed to save config to '%s': %s", file_path, exc)
            return False

