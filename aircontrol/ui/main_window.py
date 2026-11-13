"""Main desktop application window for AirControl.

Features:
- Live high-performance video preview with HUD overlay
- Real-time gesture state, action dispatch, and FPS telemetry cards
- Control toggle (Fist hold or button click)
- Settings panel and Hand Calibration wizard triggers
- Smooth minimize-to-system-tray workflow
"""

from __future__ import annotations
import math
import time
from typing import Optional, Dict
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
from aircontrol.utils.synthetic import create_synthetic_hand

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

        # Virtual / Default camera state
        self._virtual_gesture_mode = "auto"
        self._demo_start_time = time.time()
        self._demo_mode = False
        self._gesture_buttons: Dict[str, QPushButton] = {}

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
        main_layout.setContentsMargins(18, 14, 18, 14)
        main_layout.setSpacing(10)

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

        # Camera Source Status & Switch Deck
        cam_bar = QFrame()
        cam_bar.setStyleSheet("background-color: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 2px;")
        cam_bar_layout = QHBoxLayout(cam_bar)
        cam_bar_layout.setContentsMargins(12, 6, 12, 6)

        self.cam_status_lbl = QLabel("📷 Camera: Default Virtual Camera (Ready - No webcam required)")
        self.cam_status_lbl.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        cam_bar_layout.addWidget(self.cam_status_lbl)

        cam_bar_layout.addStretch()

        self.cam_toggle_btn = QPushButton("Try Physical Webcam")
        self.cam_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.cam_toggle_btn.setStyleSheet("""
            QPushButton { background-color: #1e293b; color: #cbd5e1; padding: 4px 12px; border-radius: 4px; border: 1px solid #334155; font-size: 11px; }
            QPushButton:hover { background-color: #334155; color: white; }
        """)
        self.cam_toggle_btn.clicked.connect(self._toggle_camera_source)
        cam_bar_layout.addWidget(self.cam_toggle_btn)

        main_layout.addWidget(cam_bar)

        # Video Preview Container
        preview_container = QFrame()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(4, 4, 4, 4)

        self.video_label = QLabel("Initializing camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 6px;")
        preview_layout.addWidget(self.video_label)

        main_layout.addWidget(preview_container, stretch=1)

        # Interactive Gesture Quick Selector Deck
        gesture_deck = QFrame()
        gesture_deck.setStyleSheet("background-color: #111827; border: 1px solid #1e293b; border-radius: 6px;")
        gesture_deck_layout = QHBoxLayout(gesture_deck)
        gesture_deck_layout.setContentsMargins(8, 6, 8, 6)
        gesture_deck_layout.setSpacing(6)

        deck_title = QLabel("Gesture Mode:")
        deck_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #64748b;")
        gesture_deck_layout.addWidget(deck_title)

        modes = [
            ("auto", "🔄 Auto Cycle"),
            ("point", "☝️ Move Cursor"),
            ("pinch", "👌 Pinch Click"),
            ("scroll", "✌️ Scroll"),
            ("swipe", "↔️ Swipe Nav"),
            ("volume", "🔊 Swipe Volume"),
            ("fist", "✊ Fist (Toggle)"),
        ]

        self._gesture_buttons = {}
        for mode_key, mode_label in modes:
            btn = QPushButton(mode_label)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, m=mode_key: self._set_virtual_gesture_mode(m))
            gesture_deck_layout.addWidget(btn)
            self._gesture_buttons[mode_key] = btn

        self._update_gesture_button_styles()
        main_layout.addWidget(gesture_deck)

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

    def _set_virtual_gesture_mode(self, mode: str) -> None:
        self._virtual_gesture_mode = mode
        self._demo_start_time = time.time()
        self._update_gesture_button_styles()
        logger.info("Virtual gesture mode set to '%s'.", mode)

    def _update_gesture_button_styles(self) -> None:
        for mode_key, btn in self._gesture_buttons.items():
            if mode_key == self._virtual_gesture_mode:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #0284c7; color: white; padding: 5px 12px;
                        border-radius: 4px; border: 1px solid #38bdf8; font-weight: bold; font-size: 11px;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #1e293b; color: #94a3b8; padding: 5px 12px;
                        border-radius: 4px; border: 1px solid #334155; font-size: 11px;
                    }
                    QPushButton:hover { background-color: #334155; color: white; }
                """)

    def _toggle_camera_source(self) -> None:
        if self.camera.is_default_camera:
            success = self.camera.retry_hardware_camera()
            if success:
                self.cam_status_lbl.setText("📷 Camera: Physical Hardware Webcam (Connected)")
                self.cam_status_lbl.setStyleSheet("color: #22c55e; font-weight: bold; font-size: 12px;")
                self.cam_toggle_btn.setText("Use Default Camera")
                self.tray_icon.show_notification("AirControl", "Connected to hardware webcam.")
            else:
                self.cam_status_lbl.setText("📷 Camera: Default Virtual Camera (Physical webcam unavailable)")
                self.cam_status_lbl.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
                QMessageBox.information(
                    self,
                    "Webcam Not Detected",
                    "Could not open hardware webcam index 0.\n\n"
                    "AirControl is using the Default Virtual Camera so you can test all gestures.\n\n"
                    "If you have a physical webcam:\n"
                    "• Slide open any physical camera shutter above the screen\n"
                    "• Check your keyboard camera hotkey (Fn + F8/F10)\n"
                    "• Check Windows Settings > Privacy & security > Camera"
                )
        else:
            self.camera.switch_to_default_camera()
            self.cam_status_lbl.setText("📷 Camera: Default Virtual Camera (Active)")
            self.cam_status_lbl.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
            self.cam_toggle_btn.setText("Try Physical Webcam")

    def _process_frame_tick(self) -> None:
        """Periodic tick reading latest frame, running inference, and updating display."""
        cam_frame = self.camera.get_latest_frame()

        is_default = (
            self.camera.is_default_camera
            or (cam_frame is not None and getattr(cam_frame, "is_default_camera", False))
            or not self.camera.is_connected
            or self._demo_mode
        )

        if is_default:
            now = time.time()
            t = (now - self._demo_start_time)

            if cam_frame is not None:
                frame = cam_frame.frame.copy()
            else:
                frame = np.full((480, 640, 3), 18, dtype=np.uint8)
                frame[::40, :, :] = 32
                frame[:, ::40, :] = 32

            mode = self._virtual_gesture_mode
            if mode == "auto":
                cycle_t = t % 21.0
                if cycle_t < 4.0:
                    effective_mode = "point"
                    sub_t = cycle_t
                elif cycle_t < 7.5:
                    effective_mode = "pinch"
                    sub_t = cycle_t - 4.0
                elif cycle_t < 11.0:
                    effective_mode = "scroll"
                    sub_t = cycle_t - 7.5
                elif cycle_t < 14.5:
                    effective_mode = "swipe"
                    sub_t = cycle_t - 11.0
                elif cycle_t < 17.5:
                    effective_mode = "volume"
                    sub_t = cycle_t - 14.5
                else:
                    effective_mode = "fist"
                    sub_t = cycle_t - 17.5
            else:
                effective_mode = mode
                sub_t = t

            pinch_active = False

            if effective_mode == "point":
                # Index finger up, smooth figure-8 / oval motion inside active area
                ext = (False, True, False, False, False)
                wrist_x = 0.50 + 0.16 * math.cos(sub_t * 1.5)
                wrist_y = 0.55 + 0.12 * math.sin(sub_t * 2.0)

            elif effective_mode == "pinch":
                # Periodic pinch and release
                ext = (False, False, False, False, False)
                pinch_cycle = sub_t % 1.5
                pinch_active = (pinch_cycle < 0.9)
                wrist_x = 0.50 + 0.05 * math.cos(sub_t)
                wrist_y = 0.55 + 0.05 * math.sin(sub_t)

            elif effective_mode == "scroll":
                # Two fingers up, moving vertically
                ext = (False, True, True, False, False)
                wrist_x = 0.50
                wrist_y = 0.55 + 0.15 * math.sin(sub_t * 2.5)

            elif effective_mode == "swipe":
                # Open palm, horizontal fast swipe every 1.6s
                ext = (True, True, True, True, True)
                swipe_cycle = sub_t % 1.6
                if swipe_cycle < 0.45:
                    wrist_x = 0.25 + 0.50 * (swipe_cycle / 0.45)
                else:
                    wrist_x = 0.50
                wrist_y = 0.55

            elif effective_mode == "volume":
                # Open palm, vertical fast swipe every 1.6s
                ext = (True, True, True, True, True)
                vol_cycle = sub_t % 1.6
                if vol_cycle < 0.45:
                    wrist_y = 0.75 - 0.45 * (vol_cycle / 0.45)
                else:
                    wrist_y = 0.55
                wrist_x = 0.50

            elif effective_mode == "fist":
                # All fingers curled, steady hold
                ext = (False, False, False, False, False)
                wrist_x = 0.50
                wrist_y = 0.60

            else:
                ext = (True, True, True, True, True)
                wrist_x = 0.50
                wrist_y = 0.55

            dominant_hand = create_synthetic_hand(
                wrist=(wrist_x, wrist_y),
                hand_span=0.20,
                finger_extensions=ext,
                pinch=pinch_active,
            )
            track_res_inf_time = 4.2
            cam_fps = cam_frame.fps if cam_frame else 30.0
            is_low_light = False

            # Draw visual badge on frame indicating default virtual camera
            cv2.putText(
                frame,
                "DEFAULT CAMERA (VIRTUAL FEED)",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (240, 180, 50),
                1,
                cv2.LINE_AA,
            )

        else:
            # Physical camera stream
            if cam_frame is None:
                return
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

