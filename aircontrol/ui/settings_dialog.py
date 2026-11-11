"""Settings and configuration dialog for AirControl.

Provides intuitive controls (sliders, toggles, dropdowns) to adjust:
- Mouse smoothing alpha, deadzone, and speed multiplier
- Pinch sensitivity and swipe velocity thresholds
- Fist hold arming duration
- Camera device index and horizontal mirroring
- Dominant hand selection (Right vs Left)
- Landmark skeleton and HUD display toggles
"""

from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QSlider,
    QCheckBox,
    QComboBox,
    QPushButton,
    QGroupBox,
)

from aircontrol.core.config import AppConfig
from aircontrol.utils.logger import get_logger

logger = get_logger("settings_dialog")


class SettingsDialog(QDialog):
    """Configuration panel allowing real-time sensitivity and hardware tuning."""

    config_updated = Signal(AppConfig)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("AirControl - Settings & Preferences")
        self.setFixedSize(540, 520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._init_ui()
        self._load_values()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #334155; border-radius: 6px; background-color: #0f172a; }
            QTabBar::tab { background-color: #1e293b; color: #94a3b8; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #0284c7; color: white; font-weight: bold; }
        """)

        # Tab 1: Mouse & Cursor
        tab_mouse = QWidget()
        t1_layout = QVBoxLayout(tab_mouse)
        t1_layout.setSpacing(14)

        # Smoothing Alpha
        self.alpha_slider, self.alpha_label = self._create_slider_row(
            "Cursor Smoothing (Alpha):", 5, 95, 35, "Higher = snappier, Lower = smoother"
        )
        t1_layout.addWidget(self.alpha_label)
        t1_layout.addWidget(self.alpha_slider)

        # Deadzone
        self.deadzone_slider, self.deadzone_label = self._create_slider_row(
            "Deadzone (Jitter Suppress):", 0, 15, 3, "Minimum pixel shift before cursor moves"
        )
        t1_layout.addWidget(self.deadzone_label)
        t1_layout.addWidget(self.deadzone_slider)

        # Speed Multiplier
        self.speed_slider, self.speed_label = self._create_slider_row(
            "Speed Multiplier:", 5, 25, 10, "Cursor sensitivity multiplier"
        )
        t1_layout.addWidget(self.speed_label)
        t1_layout.addWidget(self.speed_slider)

        t1_layout.addStretch()
        tabs.addTab(tab_mouse, "Cursor")

        # Tab 2: Gestures & Swipes
        tab_gestures = QWidget()
        t2_layout = QVBoxLayout(tab_gestures)
        t2_layout.setSpacing(14)

        # Fist Hold Duration
        self.fist_slider, self.fist_label = self._create_slider_row(
            "Fist Hold Toggle Duration:", 5, 25, 10, "Seconds required to toggle control ON/OFF"
        )
        t2_layout.addWidget(self.fist_label)
        t2_layout.addWidget(self.fist_slider)

        # Swipe Min Velocity
        self.swipe_vel_slider, self.swipe_vel_label = self._create_slider_row(
            "Swipe Velocity Sensitivity:", 20, 80, 45, "Higher = requires faster arm flick"
        )
        t2_layout.addWidget(self.swipe_vel_label)
        t2_layout.addWidget(self.swipe_vel_slider)

        # Pinch Threshold
        self.pinch_slider, self.pinch_label = self._create_slider_row(
            "Pinch Distance Threshold:", 30, 80, 55, "Thumb-to-index distance for click"
        )
        t2_layout.addWidget(self.pinch_label)
        t2_layout.addWidget(self.pinch_slider)

        t2_layout.addStretch()
        tabs.addTab(tab_gestures, "Gestures")

        # Tab 3: Camera & Display
        tab_cam = QWidget()
        t3_layout = QVBoxLayout(tab_cam)
        t3_layout.setSpacing(14)

        # Camera selection
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera Device:"))
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["Camera 0 (Default)", "Camera 1", "Camera 2"])
        self.cam_combo.setStyleSheet("background-color: #1e293b; color: white; padding: 4px;")
        cam_row.addWidget(self.cam_combo)
        t3_layout.addLayout(cam_row)

        # Dominant Hand
        hand_row = QHBoxLayout()
        hand_row.addWidget(QLabel("Dominant Hand:"))
        self.hand_combo = QComboBox()
        self.hand_combo.addItems(["Right Hand", "Left Hand"])
        self.hand_combo.setStyleSheet("background-color: #1e293b; color: white; padding: 4px;")
        hand_row.addWidget(self.hand_combo)
        t3_layout.addLayout(hand_row)

        # Checkboxes
        self.mirror_check = QCheckBox("Mirror Video (Natural Webcam View)")
        self.mirror_check.setStyleSheet("color: #e2e8f0;")
        t3_layout.addWidget(self.mirror_check)

        self.skeleton_check = QCheckBox("Render Hand Landmark Skeleton")
        self.skeleton_check.setStyleSheet("color: #e2e8f0;")
        t3_layout.addWidget(self.skeleton_check)

        self.box_check = QCheckBox("Show Active Control Region Box")
        self.box_check.setStyleSheet("color: #e2e8f0;")
        t3_layout.addWidget(self.box_check)

        self.hud_check = QCheckBox("Display On-Screen HUD & FPS Counter")
        self.hud_check.setStyleSheet("color: #e2e8f0;")
        t3_layout.addWidget(self.hud_check)

        t3_layout.addStretch()
        tabs.addTab(tab_cam, "Camera & HUD")

        layout.addWidget(tabs)

        # Action Buttons (Reset, Save, Cancel)
        btn_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.setStyleSheet("""
            QPushButton { background-color: #475569; color: white; padding: 6px 14px; border-radius: 4px; border: none; }
            QPushButton:hover { background-color: #64748b; }
        """)
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.reset_btn)

        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton { background-color: #334155; color: white; padding: 6px 14px; border-radius: 4px; border: none; }
            QPushButton:hover { background-color: #475569; }
        """)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.save_btn = QPushButton("Save & Apply")
        self.save_btn.setStyleSheet("""
            QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px; border: none; }
            QPushButton:hover { background-color: #0369a1; }
        """)
        self.save_btn.clicked.connect(self._save_and_apply)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

    def _create_slider_row(self, title: str, min_v: int, max_v: int, default: int, hint: str):
        label = QLabel(f"{title} ({hint})")
        label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default)
        slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: #334155; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #38bdf8; border-radius: 3px; }
            QSlider::handle:horizontal { background: #f8fafc; border: 1px solid #0284c7; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; }
        """)
        return slider, label

    def _load_values(self) -> None:
        cfg = self.config
        self.alpha_slider.setValue(int(cfg.mouse.smoothing_alpha * 100))
        self.deadzone_slider.setValue(int(cfg.mouse.deadzone_pixels))
        self.speed_slider.setValue(int(cfg.mouse.speed_multiplier * 10))

        self.fist_slider.setValue(int(cfg.gestures.fist_hold_duration_s * 10))
        self.swipe_vel_slider.setValue(int(cfg.gestures.swipe_min_velocity * 100))
        self.pinch_slider.setValue(int(cfg.gestures.pinch_distance_ratio * 1000))

        self.cam_combo.setCurrentIndex(min(2, max(0, cfg.camera.camera_index)))
        self.hand_combo.setCurrentIndex(0 if cfg.preferences.dominant_hand.lower().startswith("r") else 1)

        self.mirror_check.setChecked(cfg.camera.flip_horizontal)
        self.skeleton_check.setChecked(cfg.preferences.show_landmark_skeleton)
        self.box_check.setChecked(cfg.preferences.show_active_region_box)
        self.hud_check.setChecked(cfg.preferences.show_hud_stats)

    def _save_and_apply(self) -> None:
        cfg = self.config
        cfg.mouse.smoothing_alpha = self.alpha_slider.value() / 100.0
        cfg.mouse.deadzone_pixels = float(self.deadzone_slider.value())
        cfg.mouse.speed_multiplier = self.speed_slider.value() / 10.0

        cfg.gestures.fist_hold_duration_s = self.fist_slider.value() / 10.0
        cfg.gestures.swipe_min_velocity = self.swipe_vel_slider.value() / 100.0
        cfg.gestures.pinch_distance_ratio = self.pinch_slider.value() / 1000.0

        cfg.camera.camera_index = self.cam_combo.currentIndex()
        cfg.preferences.dominant_hand = "Right" if self.hand_combo.currentIndex() == 0 else "Left"

        cfg.camera.flip_horizontal = self.mirror_check.isChecked()
        cfg.preferences.show_landmark_skeleton = self.skeleton_check.isChecked()
        cfg.preferences.show_active_region_box = self.box_check.isChecked()
        cfg.preferences.show_hud_stats = self.hud_check.isChecked()

        cfg.save()
        self.config_updated.emit(cfg)
        logger.info("Settings updated and saved to config.json.")
        self.accept()

    def _reset_defaults(self) -> None:
        default_cfg = AppConfig()
        self.config.mouse = default_cfg.mouse
        self.config.gestures = default_cfg.gestures
        self.config.preferences = default_cfg.preferences
        self._load_values()

