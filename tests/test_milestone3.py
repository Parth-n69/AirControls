"""Unit tests for Milestone 3: Mouse and Keyboard Action Controllers."""

import time
import pytest

from aircontrol.core.config import MouseConfig, ActiveRegionConfig
from aircontrol.actions.mouse import MouseController, MockMouseBackend
from aircontrol.actions.keyboard import KeyboardController, MockKeyboardBackend


@pytest.fixture
def mock_mouse_setup():
    mouse_cfg = MouseConfig(
        smoothing_alpha=0.5,
        deadzone_pixels=4.0,
        drag_hold_delay_s=0.20,
        scroll_factor=1.0,
        scroll_deadzone=0.02,
    )
    active_reg = ActiveRegionConfig(0.2, 0.2, 0.8, 0.8)
    backend = MockMouseBackend(screen_size=(1920, 1080))
    controller = MouseController(
        mouse_config=mouse_cfg,
        active_region=active_reg,
        backend=backend,
        flip_horizontal=False,
    )
    return controller, backend


def test_mouse_cursor_movement_and_smoothing(mock_mouse_setup):
    controller, backend = mock_mouse_setup

    
    sx, sy = controller.process_point(0.5, 0.5)
    assert sx == pytest.approx(960, abs=2)
    assert sy == pytest.approx(540, abs=2)
    assert backend.cursor_pos == (sx, sy)

    # Micro-movement below deadzone should be suppressed
    sx2, sy2 = controller.process_point(0.501, 0.501)
    assert sx2 == sx
    assert sy2 == sy


def test_mouse_quick_pinch_triggers_click(mock_mouse_setup):
    controller, backend = mock_mouse_setup
    assert backend.clicks_count == 0
    assert not backend.is_down

    t0 = 100.0
    # Start pinch
    controller.handle_pinch(is_pinching=True, timestamp=t0)
    assert not backend.is_down  # Not yet dragging

    # Release pinch quickly (< 0.20s drag threshold)
    controller.handle_pinch(is_pinching=False, timestamp=t0 + 0.10)
    assert backend.clicks_count == 1
    assert not backend.is_down


def test_mouse_pinch_and_hold_triggers_drag(mock_mouse_setup):
    controller, backend = mock_mouse_setup
    assert not backend.is_down

    t0 = 200.0
    # Start pinch
    controller.handle_pinch(is_pinching=True, timestamp=t0)

    # Hold pinch past drag_hold_delay_s (0.20s)
    controller.handle_pinch(is_pinching=True, current_pos=(0.6, 0.6), timestamp=t0 + 0.25)
    assert backend.is_down  # Mouse down engaged!
    assert backend.clicks_count == 0

    # Release pinch -> mouse up
    controller.handle_pinch(is_pinching=False, timestamp=t0 + 0.40)
    assert not backend.is_down
    assert backend.clicks_count == 0  # Drag release does NOT fire a click


def test_mouse_two_finger_scroll(mock_mouse_setup):
    controller, backend = mock_mouse_setup

    # First point initializes reference position
    assert controller.process_scroll(0.5) == 0

    # Small movement below deadzone should not scroll
    assert controller.process_scroll(0.505) == 0
    assert backend.scroll_total == 0

    # Significant movement UP (decreasing Y, e.g. from 0.5 to 0.4 = -0.1 delta)
    clicks = controller.process_scroll(0.40)
    assert clicks > 0
    assert backend.scroll_total > 0


def test_keyboard_controller_navigation_and_debounce():
    mock_backend = MockKeyboardBackend()
    keyboard = KeyboardController(backend=mock_backend, min_key_interval_s=0.20)

    # Trigger Left
    assert keyboard.trigger_left() is True
    assert mock_backend.pressed_keys == ["left"]

    # Immediate second press should be debounced
    assert keyboard.trigger_left() is False
    assert mock_backend.pressed_keys == ["left"]

    # Trigger Right after debounce interval
    time.sleep(0.22)
    assert keyboard.trigger_right() is True
    assert mock_backend.pressed_keys == ["left", "right"]
