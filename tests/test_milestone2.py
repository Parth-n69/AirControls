"""Unit tests for Milestone 2: Gesture Classifier, Dynamic Swipe Detector, and FSM."""

import pytest
from tests.conftest import create_synthetic_hand
from aircontrol.core.config import GestureThresholds
from aircontrol.core.classifier import GestureClassifier, StaticGesture
from aircontrol.core.swipe import SwipeDetector, SwipeDirection
from aircontrol.core.state import ControlStateMachine, ControlState


@pytest.fixture
def default_thresholds():
    return GestureThresholds()


def test_classify_fist(default_thresholds):
    classifier = GestureClassifier(default_thresholds)
    # All fingers curled
    fist_hand = create_synthetic_hand(finger_extensions=(False, False, False, False, False))
    result = classifier.classify(fist_hand)
    assert result.gesture == StaticGesture.FIST
    assert result.confidence >= 0.90


def test_classify_point(default_thresholds):
    classifier = GestureClassifier(default_thresholds)
    # Only index extended
    point_hand = create_synthetic_hand(finger_extensions=(False, True, False, False, False))
    result = classifier.classify(point_hand)
    assert result.gesture == StaticGesture.POINT


def test_classify_two_fingers_up(default_thresholds):
    classifier = GestureClassifier(default_thresholds)
    # Index + Middle extended
    two_hand = create_synthetic_hand(finger_extensions=(False, True, True, False, False))
    result = classifier.classify(two_hand)
    assert result.gesture == StaticGesture.TWO_FINGERS_UP


def test_classify_three_fingers_up(default_thresholds):
    classifier = GestureClassifier(default_thresholds)
    # Index + Middle + Ring extended
    three_hand = create_synthetic_hand(finger_extensions=(False, True, True, True, False))
    result = classifier.classify(three_hand)
    assert result.gesture == StaticGesture.THREE_FINGERS_UP


def test_classify_open_palm(default_thresholds):
    classifier = GestureClassifier(default_thresholds)
    # All five fingers extended
    palm_hand = create_synthetic_hand(finger_extensions=(True, True, True, True, True))
    result = classifier.classify(palm_hand)
    assert result.gesture == StaticGesture.OPEN_PALM


def test_classify_pinch(default_thresholds):
    classifier = GestureClassifier(default_thresholds)
    # Pinch forced: thumb tip and index tip close
    pinch_hand = create_synthetic_hand(finger_extensions=(False, False, False, False, False), pinch=True)
    result = classifier.classify(pinch_hand)
    assert result.gesture == StaticGesture.PINCH


def test_classifier_scale_invariance(default_thresholds):
    classifier = GestureClassifier(default_thresholds)
    # 2x enlarged hand and 0.5x smaller hand should classify identically
    large_hand = create_synthetic_hand(finger_extensions=(False, True, False, False, False), scale=2.0)
    small_hand = create_synthetic_hand(finger_extensions=(False, True, False, False, False), scale=0.5)

    assert classifier.classify(large_hand).gesture == StaticGesture.POINT
    assert classifier.classify(small_hand).gesture == StaticGesture.POINT


def test_swipe_detector_horizontal(default_thresholds):
    swipe_detector = SwipeDetector(default_thresholds, flip_horizontal=False)

    # Simulate fast rightward motion with OPEN_PALM over 120ms
    t0 = 100.0
    swipe_detector.update((0.20, 0.50), StaticGesture.OPEN_PALM, timestamp=t0)
    swipe_detector.update((0.30, 0.51), StaticGesture.OPEN_PALM, timestamp=t0 + 0.05)
    ev = swipe_detector.update((0.45, 0.50), StaticGesture.OPEN_PALM, timestamp=t0 + 0.12)

    assert ev is not None
    assert ev.direction == SwipeDirection.SWIPE_RIGHT

    # Cooldown should prevent immediate re-trigger
    ev_cooldown = swipe_detector.update((0.60, 0.50), StaticGesture.OPEN_PALM, timestamp=t0 + 0.20)
    assert ev_cooldown is None


def test_swipe_detector_vertical(default_thresholds):
    swipe_detector = SwipeDetector(default_thresholds, flip_horizontal=False)

    # Simulate fast upward motion (-Y)
    t0 = 200.0
    swipe_detector.update((0.50, 0.60), StaticGesture.OPEN_PALM, timestamp=t0)
    swipe_detector.update((0.51, 0.45), StaticGesture.OPEN_PALM, timestamp=t0 + 0.05)
    ev = swipe_detector.update((0.50, 0.30), StaticGesture.OPEN_PALM, timestamp=t0 + 0.12)

    assert ev is not None
    assert ev.direction == SwipeDirection.SWIPE_UP


def test_swipe_direction_lock_rejects_diagonal(default_thresholds):
    swipe_detector = SwipeDetector(default_thresholds, flip_horizontal=False)

    # Pure diagonal motion: dx ~ dy, should fail direction lock ratio
    t0 = 300.0
    swipe_detector.update((0.20, 0.20), StaticGesture.OPEN_PALM, timestamp=t0)
    swipe_detector.update((0.30, 0.30), StaticGesture.OPEN_PALM, timestamp=t0 + 0.05)
    ev = swipe_detector.update((0.45, 0.45), StaticGesture.OPEN_PALM, timestamp=t0 + 0.12)

    assert ev is None


def test_state_machine_fist_arming_and_toggle(default_thresholds):
    fsm = ControlStateMachine(default_thresholds)
    assert fsm.current_state == ControlState.IDLE
    assert not fsm.is_active

    # Step 1: Start holding fist
    t = 10.0
    fsm.update(StaticGesture.FIST, timestamp=t)
    snap = fsm.update(StaticGesture.FIST, timestamp=t + 0.1)
    assert snap.control_state == ControlState.ARMED
    assert snap.fist_hold_progress > 0.0

    # Step 2: Release fist at 0.5s -> should revert to IDLE
    snap_rel = fsm.update(StaticGesture.POINT, timestamp=t + 0.5)
    assert snap_rel.control_state == ControlState.IDLE
    assert snap_rel.fist_hold_progress == 0.0

    # Step 3: Hold fist continuously for full 1.0s -> should toggle to ACTIVE!
    t2 = 20.0
    fsm.update(StaticGesture.FIST, timestamp=t2)
    fsm.update(StaticGesture.FIST, timestamp=t2 + 0.5)
    snap_act = fsm.update(StaticGesture.FIST, timestamp=t2 + 1.05)

    assert snap_act.control_state == ControlState.ACTIVE
    assert fsm.is_active

    # Step 4: Hold fist while ACTIVE for 1.0s -> should toggle back to IDLE
    t3 = 30.0
    fsm.update(StaticGesture.FIST, timestamp=t3)
    snap_idle = fsm.update(StaticGesture.FIST, timestamp=t3 + 1.05)
    assert snap_idle.control_state == ControlState.IDLE
    assert not fsm.is_active


def test_state_machine_frame_stabilization(default_thresholds):
    fsm = ControlStateMachine(default_thresholds)
    fsm.force_toggle()  # Now ACTIVE

    # Consecutive frames required: 3
    fsm.update(StaticGesture.POINT)
    assert fsm.stable_gesture != StaticGesture.POINT  # only 1 frame
    fsm.update(StaticGesture.POINT)
    assert fsm.stable_gesture != StaticGesture.POINT  # 2 frames
    fsm.update(StaticGesture.POINT)
    assert fsm.stable_gesture == StaticGesture.POINT  # 3 frames reached!

    # Rogue 1-frame flicker should NOT dislodge POINT
    fsm.update(StaticGesture.UNKNOWN)
    assert fsm.stable_gesture == StaticGesture.POINT
