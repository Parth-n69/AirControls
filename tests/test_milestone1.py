"""Unit tests for Milestone 1: Math Helpers, Config, Camera, Tracker, and Overlay."""

import os
import tempfile
import numpy as np
import pytest

from aircontrol.utils.math_helpers import (
    euclidean_distance_2d,
    euclidean_distance_3d,
    clamp,
    map_active_region_to_screen,
    ExponentialSmoothing2D,
)
from aircontrol.core.config import AppConfig, ActiveRegionConfig
from aircontrol.core.tracker import HandTracker, HandLandmark, HandData, WRIST, INDEX_TIP
from aircontrol.ui.overlay import OverlayRenderer


def test_math_helpers_distances():
    assert euclidean_distance_2d((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert euclidean_distance_3d((0.0, 0.0, 0.0), (1.0, 2.0, 2.0)) == pytest.approx(3.0)
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(-2.0, 0.0, 10.0) == 0.0
    assert clamp(15.0, 0.0, 10.0) == 10.0


def test_active_region_mapping():
    active_box = (0.2, 0.2, 0.8, 0.8)
    screen_w, screen_h = 1920, 1080

    # Test center with flip_horizontal=False
    sx, sy = map_active_region_to_screen(
        norm_x=0.5, norm_y=0.5,
        active_box=active_box,
        screen_width=screen_w, screen_height=screen_h,
        flip_horizontal=False
    )
    assert sx == pytest.approx(screen_w / 2, abs=2)
    assert sy == pytest.approx(screen_h / 2, abs=2)

    # Test clamping outside active box
    sx_out, sy_out = map_active_region_to_screen(
        norm_x=0.0, norm_y=1.0,
        active_box=active_box,
        screen_width=screen_w, screen_height=screen_h,
        flip_horizontal=False
    )
    assert sx_out == 0
    assert sy_out == screen_h - 1


def test_exponential_smoothing_and_deadzone():
    smoother = ExponentialSmoothing2D(alpha=0.5, deadzone=5.0)
    assert smoother.current_value is None

    # First update initializes
    x1, y1 = smoother.update(100.0, 100.0)
    assert (x1, y1) == (100.0, 100.0)

    # Micro-movement below deadzone should be suppressed
    x2, y2 = smoother.update(102.0, 101.0)
    assert (x2, y2) == (100.0, 100.0)

    # Significant movement should be smoothed: 0.5 * 200 + 0.5 * 100 = 150
    x3, y3 = smoother.update(200.0, 200.0)
    assert (x3, y3) == (150.0, 150.0)

    # Reset
    smoother.reset()
    assert smoother.current_value is None


def test_config_load_and_save():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "test_config.json")
        cfg = AppConfig.load(config_path)

        assert cfg.camera.camera_index == 0
        assert cfg.gestures.fist_hold_duration_s == 1.0

        # Modify and save
        cfg.preferences.dominant_hand = "Left"
        cfg.mouse.smoothing_alpha = 0.50
        assert cfg.save(config_path) is True

        # Reload
        reloaded = AppConfig.load(config_path)
        assert reloaded.preferences.dominant_hand == "Left"
        assert reloaded.mouse.smoothing_alpha == 0.50


def test_tracker_on_blank_frame(blank_frame):
    tracker = HandTracker(
        model_complexity=0,
        inference_resolution=(320, 180),
        dominant_hand_preference="Right",
    )
    try:
        result = tracker.process(blank_frame)
        assert result is not None
        assert len(result.hands) == 0
        assert result.dominant_hand is None
        assert result.inference_time_ms >= 0.0
    finally:
        tracker.close()


def test_overlay_rendering(blank_frame):
    renderer = OverlayRenderer()
    active_box = ActiveRegionConfig(0.2, 0.2, 0.8, 0.8)

    # Draw active region box
    frame_copy = blank_frame.copy()
    renderer.draw_active_region_box(frame_copy, active_box)
    assert np.any(frame_copy > 0)

    # Draw HUD
    hud_frame = blank_frame.copy()
    renderer.draw_hud(
        hud_frame,
        state_label="ACTIVE",
        gesture_label="POINT",
        fps=30.0,
        latency_ms=12.5,
        hold_progress=0.0,
        is_low_light=True,
    )
    assert np.any(hud_frame > 0)

