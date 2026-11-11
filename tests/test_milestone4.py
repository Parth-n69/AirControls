"""Unit tests for Milestone 4: Volume Controller and Action Dispatcher Integration."""

import pytest
from tests.conftest import create_synthetic_hand
from aircontrol.core.config import AppConfig
from aircontrol.core.classifier import StaticGesture, ClassificationResult, FingerState
from aircontrol.core.swipe import SwipeDetector, SwipeEvent, SwipeDirection
from aircontrol.core.state import ControlStateMachine, StateSnapshot, ControlState
from aircontrol.actions.mouse import MouseController, MockMouseBackend
from aircontrol.actions.keyboard import KeyboardController, MockKeyboardBackend
from aircontrol.actions.volume import MockVolumeController, get_volume_controller, VolumeController
from aircontrol.core.dispatcher import ActionDispatcher


@pytest.fixture
def test_pipeline():
    config = AppConfig()
    fsm = ControlStateMachine(config.gestures)
    fsm.force_toggle()  # Make ACTIVE

    swipe_detector = SwipeDetector(config.gestures, flip_horizontal=False)
    mouse_backend = MockMouseBackend(screen_size=(1920, 1080))
    mouse = MouseController(config.mouse, config.active_region, backend=mouse_backend, flip_horizontal=False)
    keyboard_backend = MockKeyboardBackend()
    keyboard = KeyboardController(backend=keyboard_backend, min_key_interval_s=0.01)
    volume = MockVolumeController(initial_volume=0.50)

    dispatcher = ActionDispatcher(
        config=config,
        state_machine=fsm,
        swipe_detector=swipe_detector,
        mouse_controller=mouse,
        keyboard_controller=keyboard,
        volume_controller=volume,
    )
    return {
        "dispatcher": dispatcher,
        "fsm": fsm,
        "mouse_backend": mouse_backend,
        "keyboard_backend": keyboard_backend,
        "volume": volume,
        "config": config,
    }


def test_mock_volume_controller_operations():
    vol = MockVolumeController(initial_volume=0.50)
    assert vol.get_volume() == 0.50

    # Step volume up 5%
    new_v = vol.step_volume(0.05)
    assert new_v == pytest.approx(0.55)
    assert vol.get_volume() == pytest.approx(0.55)

    # Step volume down 10%
    new_v2 = vol.step_volume(-0.10)
    assert new_v2 == pytest.approx(0.45)

    # Clamping test at upper bound
    vol.set_volume(1.50)
    assert vol.get_volume() == 1.0

    # Clamping test at lower bound
    vol.step_volume(-2.0)
    assert vol.get_volume() == 0.0

    # Mute
    assert not vol.is_muted()
    vol.set_mute(True)
    assert vol.is_muted()


def test_volume_controller_factory():
    controller = get_volume_controller()
    assert isinstance(controller, VolumeController)
    assert 0.0 <= controller.get_volume() <= 1.0


def test_dispatcher_suppressed_when_idle(test_pipeline):
    dispatcher = test_pipeline["dispatcher"]
    fsm = test_pipeline["fsm"]
    fsm.force_toggle()  # Switch to IDLE

    hand = create_synthetic_hand(finger_extensions=(False, True, False, False, False))
    classif = ClassificationResult(
        gesture=StaticGesture.POINT,
        confidence=0.95,
        finger_state=FingerState(False, True, False, False, False),
        pinch_distance_ratio=0.5,
        normalized_hand_span=0.2,
    )
    snap = StateSnapshot(
        control_state=ControlState.IDLE,
        stable_gesture=StaticGesture.POINT,
        raw_gesture=StaticGesture.POINT,
        fist_hold_progress=0.0,
        is_active=False,
    )

    result = dispatcher.process_frame(hand, classif, snap)
    assert result.action_type == "IDLE"
    assert test_pipeline["mouse_backend"].cursor_pos == (960, 540)  # Unmoved


def test_dispatcher_point_moves_cursor(test_pipeline):
    dispatcher = test_pipeline["dispatcher"]
    hand = create_synthetic_hand(wrist=(0.5, 0.8), finger_extensions=(False, True, False, False, False))
    classif = ClassificationResult(
        gesture=StaticGesture.POINT,
        confidence=0.95,
        finger_state=FingerState(False, True, False, False, False),
        pinch_distance_ratio=0.5,
        normalized_hand_span=0.2,
    )
    snap = StateSnapshot(
        control_state=ControlState.ACTIVE,
        stable_gesture=StaticGesture.POINT,
        raw_gesture=StaticGesture.POINT,
        fist_hold_progress=0.0,
        is_active=True,
    )

    result = dispatcher.process_frame(hand, classif, snap)
    assert result.action_type == "MOUSE_MOVE"
    assert result.cursor_pos is not None


def test_dispatcher_three_fingers_volume_slider(test_pipeline):
    dispatcher = test_pipeline["dispatcher"]
    volume = test_pipeline["volume"]

    # Active region Y is [0.20, 0.80]. Palm at 0.20 should map to volume 1.0 (100%)
    hand_top = create_synthetic_hand(wrist=(0.5, 0.3))  # palm_center ~ 0.20
    classif = ClassificationResult(
        gesture=StaticGesture.THREE_FINGERS_UP,
        confidence=0.90,
        finger_state=FingerState(False, True, True, True, False),
        pinch_distance_ratio=0.5,
        normalized_hand_span=0.2,
    )
    snap = StateSnapshot(
        control_state=ControlState.ACTIVE,
        stable_gesture=StaticGesture.THREE_FINGERS_UP,
        raw_gesture=StaticGesture.THREE_FINGERS_UP,
        fist_hold_progress=0.0,
        is_active=True,
    )

    res = dispatcher.process_frame(hand_top, classif, snap)
    assert res.action_type == "VOLUME_SLIDER"
    assert volume.get_volume() > 0.80  # Near top = high volume


def test_dispatcher_swipe_actions(test_pipeline):
    dispatcher = test_pipeline["dispatcher"]
    keyboard = test_pipeline["keyboard_backend"]
    volume = test_pipeline["volume"]

    # Direct test of _dispatch_swipe
    # 1. Swipe Left -> Left Arrow
    ev_left = SwipeEvent(SwipeDirection.SWIPE_LEFT, -0.3, 0.0, -1.0, 0.0, 100.0)
    res_l = dispatcher._dispatch_swipe(ev_left)
    assert res_l.action_type == "KEYBOARD"
    assert keyboard.pressed_keys[-1] == "left"

    # 2. Swipe Right -> Right Arrow
    import time
    time.sleep(0.03)
    ev_right = SwipeEvent(SwipeDirection.SWIPE_RIGHT, 0.3, 0.0, 1.0, 0.0, 100.1)
    res_r = dispatcher._dispatch_swipe(ev_right)
    assert res_r.action_type == "KEYBOARD"
    assert keyboard.pressed_keys[-1] == "right"

    # 3. Swipe Up -> Volume +5%
    initial_v = volume.get_volume()
    ev_up = SwipeEvent(SwipeDirection.SWIPE_UP, 0.0, -0.3, 0.0, -1.0, 100.2)
    res_u = dispatcher._dispatch_swipe(ev_up)
    assert res_u.action_type == "VOLUME_STEP"
    assert volume.get_volume() == pytest.approx(initial_v + 0.05)

    # 4. Swipe Down -> Volume -5%
    ev_down = SwipeEvent(SwipeDirection.SWIPE_DOWN, 0.0, 0.3, 0.0, 1.0, 100.3)
    res_d = dispatcher._dispatch_swipe(ev_down)
    assert res_d.action_type == "VOLUME_STEP"
    assert volume.get_volume() == pytest.approx(initial_v)
