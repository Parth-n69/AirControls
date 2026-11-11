"""Main desktop application window for AirControl.

Features:
- Live high-performance video preview with HUD overlay
- Real-time gesture state, action dispatch, and FPS telemetry cards
- Control toggle (Fist hold or button click)
- Settings panel and Hand Calibration wizard triggers
- Smooth minimize-to-system-tray workflow
"""

from __future__ import annotations
import time
from typing import Optional
import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QMessageBox,
)

from aircontrol.core.config import AppConfig
from aircontrol.core.camera import CameraThread, CameraFrame
from aircontrol.core.tracker import HandTracker
from aircontrol.core.classifier import GestureClassifier, ClassificationResult, StaticGesture
from aircontrol.core.swipe import SwipeDetector
from aircontrol.core.state import ControlStateMachine, StateSnapshot, ControlState
from aircontrol.core.dispatcher import ActionDispatcher, DispatchResult
from aircontrol.actions.mouse import MouseController
from aircontrol.actions.keyboard import KeyboardController
from aircontrol.actions.volume import get_volume_controller
from aircontrol.ui.overlay import OverlayRenderer
from aircontrol.ui.settings_dialog import SettingsDialog
from aircontrol.ui.calibration_dialog import CalibrationDialog
from aircontrol.ui.tray_icon import AirControlTrayIcon
from aircontrol.utils.logger import get_logger

logger = get_logger("main_window")


class MainWindow(QMainWindow):
    """Primary application window displaying video feed and telemetry."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("AirControl - Touchless Hand Gesture Desktop Control")
        self.setMinimumSize(880, 680)

        # 1. Initialize Subsystems
        self.camera = CameraThread(self.config.camera)
        self.tracker = HandTracker(
            dominant_hand_preference=self.config.preferences.dominant_hand,
        )
        self.classifier = GestureClassifier(
            thresholds=self.config.gestures,
            calibration_scale=self.config.preferences.calibration_scale,
        )
        self.swipe_detector = SwipeDetector(
            thresholds=self.config.gestures,
            flip_horizontal=self.config.camera.flip_horizontal,
        )
        self.state_machine = ControlStateMachine(self.config.gestures)
        self.mouse = MouseController(
            mouse_config=self.config.mouse,
            active_region=self.config.active_region,
            flip_horizontal=self.config.camera.flip_horizontal,
        )
        self.keyboard = KeyboardController()
        self.volume = get_volume_controller()

        self.dispatcher = ActionDispatcher(
            config=self.config,
            state_machine=self.state_machine,
            swipe_detector=self.swipe_detector,
            mouse_controller=self.mouse,
            keyboard_controller=self.keyboard,
            volume_controller=self.volume,
        )
        self.overlay = OverlayRenderer()

        # Calibration Dialog reference
        self._calibration_dlg: Optional[CalibrationDialog] = None

        # Demo simulation state
        self._demo_mode = False
        self._demo_start_time = 0.0

        # 2. Build User Interface
        self._init_ui()

        # 3. Setup System Tray
        self.tray_icon = AirControlTrayIcon(
            parent=self,
            on_show=self._show_window_from_tray,
            on_toggle=self._toggle_control,
            on_exit=self._clean_exit,
        )
        self.tray_icon.show()

        # State machine change callback
        self.state_machine.register_state_change_callback(self._on_state_changed)

        # 4. Start Camera Acquisition and Main Processing Loop Timer
        self.camera.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame_tick)
        self.timer.start(15)  # ~60 Hz polling loop

    def _init_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; }
            QLabel { color: #f8fafc; font-family: 'Segoe UI', Arial, sans-serif; }
            QFrame { background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(14)

        # Header Bar
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        app_title = QLabel("AirControl")
        app_title.setStyleSheet("font-size: 22px; font-weight: 800; color: #38bdf8; letter-spacing: 0.5px;")
        app_sub = QLabel("Production Mid-Air Gesture Interface")
        app_sub.setStyleSheet("font-size: 12px; color: #94a3b8;")
        title_box.addWidget(app_title)
        title_box.addWidget(app_sub)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        # Big Toggle Button
        self.toggle_btn = QPushButton("Turn Control ON")
        self.toggle_btn.setMinimumSize(180, 42)
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self._update_toggle_button_style(is_active=False)
        self.toggle_btn.clicked.connect(self._toggle_control)
        header_layout.addWidget(self.toggle_btn)

        main_layout.addLayout(header_layout)

        # Video Preview Container
        preview_container = QFrame()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(4, 4, 4, 4)

        self.video_label = QLabel("Initializing camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 6px;")
        preview_layout.addWidget(self.video_label)

        main_layout.addWidget(preview_container, stretch=1)

        # Telemetry & Status Cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        self.card_state = self._create_telemetry_card("SYSTEM STATE", "IDLE (Hold Fist 1s)", "#94a3b8")
        self.card_gesture = self._create_telemetry_card("CURRENT GESTURE", "NONE", "#38bdf8")
        self.card_action = self._create_telemetry_card("DISPATCHED ACTION", "Ready", "#a855f7")
        self.card_fps = self._create_telemetry_card("CAMERA / LATENCY", "0.0 FPS | 0ms", "#22c55e")

        cards_layout.addWidget(self.card_state)
        cards_layout.addWidget(self.card_gesture)
        cards_layout.addWidget(self.card_action)
        cards_layout.addWidget(self.card_fps)

        main_layout.addLayout(cards_layout)

        # Quick Actions Footer
        footer_layout = QHBoxLayout()

        self.calib_btn = QPushButton("Calibrate Hand Size")
        self.calib_btn.setStyleSheet("""
            QPushButton { background-color: #1e293b; color: white; padding: 8px 16px; border-radius: 6px; border: 1px solid #334155; }
            QPushButton:hover { background-color: #334155; }
        """)
        self.calib_btn.clicked.connect(self._open_calibration)
        footer_layout.addWidget(self.calib_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setStyleSheet("""
            QPushButton { background-color: #1e293b; color: white; padding: 8px 16px; border-radius: 6px; border: 1px solid #334155; }
            QPushButton:hover { background-color: #334155; }
        """)
        self.settings_btn.clicked.connect(self._open_settings)
        footer_layout.addWidget(self.settings_btn)

        self.demo_btn = QPushButton("Simulate Hand (Demo)")
        self.demo_btn.setStyleSheet("""
            QPushButton { background-color: #0369a1; color: white; padding: 8px 16px; border-radius: 6px; border: 1px solid #0284c7; font-weight: bold; }
            QPushButton:hover { background-color: #0284c7; }
        """)
        self.demo_btn.clicked.connect(self._toggle_demo_mode)
        footer_layout.addWidget(self.demo_btn)

        footer_layout.addStretch()

        self.tray_btn = QPushButton("Minimize to Tray")
        self.tray_btn.setStyleSheet("""
            QPushButton { background-color: #1e293b; color: #94a3b8; padding: 8px 16px; border-radius: 6px; border: 1px solid #334155; }
            QPushButton:hover { background-color: #334155; color: white; }
        """)
        self.tray_btn.clicked.connect(self.hide)
        footer_layout.addWidget(self.tray_btn)

        main_layout.addLayout(footer_layout)

    def _create_telemetry_card(self, title: str, default_val: str, accent_color: str) -> QFrame:
        card = QFrame()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #64748b; letter-spacing: 0.5px;")

        val_lbl = QLabel(default_val)
        val_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {accent_color};")
        val_lbl.setObjectName("card_value")

        card_layout.addWidget(title_lbl)
        card_layout.addWidget(val_lbl)
        return card

    def _update_card(self, card: QFrame, text: str) -> None:
        val_lbl = card.findChild(QLabel, "card_value")
        if val_lbl:
            val_lbl.setText(text)

    def _update_toggle_button_style(self, is_active: bool) -> None:
        if is_active:
            self.toggle_btn.setText("Turn Control OFF")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444; color: white; font-weight: bold;
                    font-size: 14px; border-radius: 6px; border: none;
                }
                QPushButton:hover { background-color: #dc2626; }
            """)
        else:
            self.toggle_btn.setText("Turn Control ON")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #22c55e; color: #0b0f19; font-weight: bold;
                    font-size: 14px; border-radius: 6px; border: none;
                }
                QPushButton:hover { background-color: #16a34a; }
            """)

    def _toggle_control(self) -> None:
        new_state = self.state_machine.force_toggle()
        self._update_toggle_button_style(new_state == ControlState.ACTIVE)
        self.tray_icon.update_state_icon(new_state == ControlState.ACTIVE)

    def _on_state_changed(self, new_state: ControlState) -> None:
        is_act = new_state == ControlState.ACTIVE
        self._update_toggle_button_style(is_act)
        self.tray_icon.update_state_icon(is_act)
        state_name = "ACTIVE" if is_act else "IDLE"
        self.tray_icon.show_notification("AirControl", f"Gesture Control is now {state_name}.")

    def _toggle_demo_mode(self) -> None:
        self._demo_mode = not self._demo_mode
        if self._demo_mode:
            self._demo_start_time = time.time()
            self.demo_btn.setText("Stop Simulation")
            self.demo_btn.setStyleSheet("""
                QPushButton { background-color: #ea580c; color: white; padding: 8px 16px; border-radius: 6px; border: none; font-weight: bold; }
                QPushButton:hover { background-color: #c2410c; }
            """)
            logger.info("Demo simulation mode activated.")
        else:
            self.demo_btn.setText("Simulate Hand (Demo)")
            self.demo_btn.setStyleSheet("""
                QPushButton { background-color: #0369a1; color: white; padding: 8px 16px; border-radius: 6px; border: 1px solid #0284c7; font-weight: bold; }
                QPushButton:hover { background-color: #0284c7; }
            """)
            logger.info("Demo simulation mode stopped.")

    def _process_frame_tick(self) -> None:
        """Periodic tick reading latest frame, running inference, and updating display."""
        import math
        cam_frame = self.camera.get_latest_frame()

        if self._demo_mode:
            # Generate simulated frame and hand telemetry for demonstration
            now = time.time()
            t = (now - self._demo_start_time) % 20.0

            frame = np.full((480, 640, 3), 18, dtype=np.uint8)
            frame[::40, :, :] = 32
            frame[:, ::40, :] = 32

            # Smooth Lissajous motion
            wrist_x = 0.50 + 0.18 * math.cos(t * 1.5)
            wrist_y = 0.65 + 0.12 * math.sin(t * 1.5)

            pinch_active = False
            if t < 5.0:
                ext = (False, True, False, False, False)  # Point
            elif t < 8.0:
                ext = (False, False, False, False, False)
                pinch_active = True                       # Pinch
            elif t < 12.0:
                ext = (False, True, True, False, False)   # Two fingers
                wrist_y = 0.60 + 0.15 * math.sin(t * 2.0)
            elif t < 16.0:
                ext = (False, True, True, True, False)   # Three fingers
                wrist_y = 0.50 + 0.20 * math.sin(t * 2.0)
            else:
                ext = (True, True, True, True, True)     # Open palm
                wrist_x = 0.35 + 0.30 * ((t - 16.0) / 4.0)

            from aircontrol.utils.synthetic import create_synthetic_hand
            dominant_hand = create_synthetic_hand(
                wrist=(wrist_x, wrist_y),
                hand_span=0.20,
                finger_extensions=ext,
                pinch=pinch_active,
            )
            track_res_inf_time = 8.5
            cam_fps = 30.0
            is_low_light = False

        elif cam_frame is None:
            if not self.camera.is_connected:
                self.video_label.setText(
                    "📷 Integrated Camera Not Open / Unavailable\n\n"
                    "• Physical Shutter: Check the small sliding switch above your laptop screen\n"
                    "  (slide it open to reveal the camera lens).\n\n"
                    "• Keyboard Privacy Key: Press Fn + F8 (or your camera hotkey)\n"
                    "  to toggle hardware camera privacy.\n\n"
                    "• Windows Camera Settings: Open Windows Settings > Privacy & security > Camera\n"
                    "  and ensure 'Let desktop apps access your camera' is ON.\n\n"
                    "AirControl is auto-retrying every 2s...\n"
                    "(Or click 'Simulate Hand (Demo)' below to test without camera!)"
                )
            return
        else:
            now = cam_frame.timestamp
            frame = cam_frame.frame.copy()
            if self.config.camera.flip_horizontal:
                frame = cv2.flip(frame, 1)

            track_res = self.tracker.process(frame)
            dominant_hand = track_res.dominant_hand
            track_res_inf_time = track_res.inference_time_ms
            cam_fps = cam_frame.fps
            is_low_light = cam_frame.is_low_light

        # 2. Static Classification
        if dominant_hand:
            classif_res = self.classifier.classify(dominant_hand)
        else:
            classif_res = ClassificationResult(
                gesture=StaticGesture.NONE,
                confidence=0.0,
                finger_state=None,
                pinch_distance_ratio=0.0,
                normalized_hand_span=0.20,
            )

        # 3. State Machine Update
        state_snap = self.state_machine.update(classif_res.gesture, timestamp=now)

        # Feed data to calibration wizard if open
        if self._calibration_dlg and self._calibration_dlg.isVisible():
            self._calibration_dlg.feed_frame_data(dominant_hand, classif_res)

        # 4. Action Dispatching
        dispatch_res = self.dispatcher.process_frame(dominant_hand, classif_res, state_snap, timestamp=now)

        # 5. Overlays (if enabled in user preferences)
        if self.config.preferences.show_active_region_box:
            self.overlay.draw_active_region_box(frame, self.config.active_region, self.config.camera.flip_horizontal)

        if self.config.preferences.show_landmark_skeleton and dominant_hand:
            self.overlay.draw_hand_skeleton(frame, dominant_hand, is_dominant=True)

        if self.config.preferences.show_hud_stats:
            self.overlay.draw_hud(
                frame=frame,
                state_label=state_snap.control_state.value,
                gesture_label=state_snap.stable_gesture.value if state_snap.stable_gesture != StaticGesture.NONE else classif_res.gesture.value,
                fps=cam_fps,
                latency_ms=track_res_inf_time,
                hold_progress=state_snap.fist_hold_progress,
                is_low_light=is_low_light,
            )

        # 6. Update Telemetry Cards
        st_text = state_snap.control_state.value
        if state_snap.control_state == ControlState.ARMED:
            st_text = f"ARMING ({state_snap.fist_hold_progress * 100:.0f}%)"
        self._update_card(self.card_state, st_text)

        g_text = state_snap.stable_gesture.value
        if g_text == "NONE" and classif_res.gesture != StaticGesture.NONE:
            g_text = f"~{classif_res.gesture.value}"
        self._update_card(self.card_gesture, g_text)

        self._update_card(self.card_action, dispatch_res.detail)
        self._update_card(self.card_fps, f"{cam_fps:.1f} FPS | {track_res_inf_time:.1f}ms")

        # 7. Convert OpenCV BGR frame to Qt Pixmap and paint to label
        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_display.shape
        qimg = QImage(rgb_display.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to label size preserving aspect ratio
        scaled_pix = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pix)

    def _open_calibration(self) -> None:
        self._calibration_dlg = CalibrationDialog(self.config, parent=self)
        self._calibration_dlg.calibration_completed.connect(self._on_calibration_completed)
        self._calibration_dlg.exec()

    def _on_calibration_completed(self, scale: float) -> None:
        self.classifier.update_calibration_scale(scale)
        logger.info("Updated classifier calibration scale to %.2fx", scale)

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.config, parent=self)
        dlg.config_updated.connect(self._on_config_updated)
        dlg.exec()

    def _on_config_updated(self, new_cfg: AppConfig) -> None:
        self.config = new_cfg
        self.mouse.update_config(new_cfg.mouse, new_cfg.active_region)
        self.tracker.set_dominant_hand(new_cfg.preferences.dominant_hand)
        self.swipe_detector.flip_horizontal = new_cfg.camera.flip_horizontal
        self.mouse.flip_horizontal = new_cfg.camera.flip_horizontal
        logger.info("Main window synchronized with updated preferences.")

    def _show_window_from_tray(self) -> None:
        self.showNormal()
        self.activateWindow()

    def _clean_exit(self) -> None:
        self.timer.stop()
        self.camera.stop()
        self.tracker.close()
        self.mouse.reset()
        self.tray_icon.hide()
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        # If tray is visible, minimize to tray on close
        if self.tray_icon.isVisible():
            self.hide()
            self.tray_icon.show_notification(
                "AirControl", "AirControl is still running in the system tray. Right-click icon to exit."
            )
            event.ignore()
        else:
            self._clean_exit()
            event.accept()

