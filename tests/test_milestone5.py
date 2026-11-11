"""Unit tests for Milestone 5: UI Dialogs, Settings, Calibration, and Window Lifecycle."""

import os
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from tests.conftest import create_synthetic_hand
from aircontrol.core.config import AppConfig
from aircontrol.core.classifier import StaticGesture, ClassificationResult, FingerState
from aircontrol.core.state import ControlState
from aircontrol.ui.settings_dialog import SettingsDialog
from aircontrol.ui.calibration_dialog import CalibrationDialog
from aircontrol.ui.tray_icon import AirControlTrayIcon, create_tray_pixmap
from aircontrol.ui.main_window import MainWindow

# Ensure Qt can run in headless/offscreen environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_tray_pixmap_generation(qapp):
    pix = create_tray_pixmap(is_active=True)
    assert not pix.isNull()
    assert pix.width() == 32
    assert pix.height() == 32

    pix_idle = create_tray_pixmap(is_active=False)
    assert not pix_idle.isNull()


def test_settings_dialog_load_and_save(qapp):
    config = AppConfig()
    config.mouse.smoothing_alpha = 0.40
    config.mouse.deadzone_pixels = 5.0

    dlg = SettingsDialog(config)
    assert dlg.alpha_slider.value() == 40
    assert dlg.deadzone_slider.value() == 5

    # Simulate user changing slider and saving
    dlg.alpha_slider.setValue(60)
    dlg.deadzone_slider.setValue(8)

    updated_events = []
    dlg.config_updated.connect(lambda cfg: updated_events.append(cfg))

    dlg._save_and_apply()
    assert len(updated_events) == 1
    assert config.mouse.smoothing_alpha == 0.60
    assert config.mouse.deadzone_pixels == 8.0
    dlg.close()


def test_calibration_dialog_sampling(qapp):
    config = AppConfig()
    dlg = CalibrationDialog(config)
    assert not dlg._is_calibrating

    # Start calibration
    dlg._toggle_calibration()
    assert dlg._is_calibrating

    calib_results = []
    dlg.calibration_completed.connect(lambda s: calib_results.append(s))

    # Feed synthetic open palm frames
    palm_hand = create_synthetic_hand(hand_span=0.25, finger_extensions=(True, True, True, True, True))
    classif = ClassificationResult(
        gesture=StaticGesture.OPEN_PALM,
        confidence=0.95,
        finger_state=FingerState(True, True, True, True, True),
        pinch_distance_ratio=0.5,
        normalized_hand_span=0.25,
    )

    for _ in range(dlg._target_samples + 5):
        dlg.feed_frame_data(palm_hand, classif)

    assert not dlg._is_calibrating
    assert len(calib_results) == 1
    assert calib_results[0] > 1.0  # 0.25 span / 0.20 baseline = ~1.25x
    dlg.close()


def test_main_window_toggle_control(qapp):
    config = AppConfig()
    window = MainWindow(config)

    # Initial state should be IDLE
    assert window.state_machine.current_state == ControlState.IDLE

    # Click toggle button
    window._toggle_control()
    assert window.state_machine.current_state == ControlState.ACTIVE

    # Click again
    window._toggle_control()
    assert window.state_machine.current_state == ControlState.IDLE

    # Clean shutdown
    window._clean_exit()

