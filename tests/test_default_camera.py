"""Unit tests for Default Virtual Camera fallback and gesture simulator."""

import time
import numpy as np
import pytest

from aircontrol.core.config import CameraConfig, AppConfig, GestureThresholds
from aircontrol.core.camera import CameraThread, CameraFrame
from aircontrol.ui.main_window import MainWindow
from aircontrol.utils.synthetic import create_synthetic_hand
from aircontrol.core.classifier import GestureClassifier, StaticGesture
from aircontrol.core.state import ControlStateMachine, ControlState


def test_camera_thread_default_virtual_fallback():
    """Verify that CameraThread automatically produces frames via default fallback."""
    cfg = CameraConfig(camera_index=-1, use_default_fallback=True, target_fps=30)
    cam = CameraThread(cfg)
    started = cam.start()
    assert started is True

    try:
        # Give thread a brief moment to produce frame
        time.sleep(0.15)
        frame = cam.get_latest_frame()
        assert frame is not None
        assert frame.is_default_camera is True
        assert frame.frame.shape == (480, 640, 3)
        assert cam.is_default_camera is True
        assert cam.is_connected is True
    finally:
        cam.stop()


def test_synthetic_hand_gestures_classification():
    """Verify that all virtual gesture modes correctly classify to their target gestures."""
    classifier = GestureClassifier(thresholds=GestureThresholds())

    # 1. Point
    hand_point = create_synthetic_hand(
        wrist=(0.5, 0.6),
        finger_extensions=(False, True, False, False, False),
        pinch=False,
    )
    res_point = classifier.classify(hand_point)
    assert res_point.gesture == StaticGesture.POINT

    # 2. Pinch
    hand_pinch = create_synthetic_hand(
        wrist=(0.5, 0.6),
        finger_extensions=(False, False, False, False, False),
        pinch=True,
    )
    res_pinch = classifier.classify(hand_pinch)
    assert res_pinch.gesture == StaticGesture.PINCH

    # 3. Two fingers up
    hand_two = create_synthetic_hand(
        wrist=(0.5, 0.6),
        finger_extensions=(False, True, True, False, False),
        pinch=False,
    )
    res_two = classifier.classify(hand_two)
    assert res_two.gesture == StaticGesture.TWO_FINGERS_UP

    # 4. Open palm
    hand_palm = create_synthetic_hand(
        wrist=(0.5, 0.6),
        finger_extensions=(True, True, True, True, True),
        pinch=False,
    )
    res_palm = classifier.classify(hand_palm)
    assert res_palm.gesture == StaticGesture.OPEN_PALM

    # 5. Fist
    hand_fist = create_synthetic_hand(
        wrist=(0.5, 0.6),
        finger_extensions=(False, False, False, False, False),
        pinch=False,
    )
    res_fist = classifier.classify(hand_fist)
    assert res_fist.gesture == StaticGesture.FIST
