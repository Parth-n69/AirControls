"""Hand calibration wizard for AirControl.

Guides the user to hold an open palm at their preferred working distance from the camera,
measures empirical hand span across a multi-frame sample, and auto-scales gesture
thresholds to adapt to hand size and webcam field-of-view.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QGroupBox,
)

from aircontrol.core.tracker import HandData
from aircontrol.core.classifier import StaticGesture, ClassificationResult
from aircontrol.core.config import AppConfig
from aircontrol.utils.logger import get_logger

logger = get_logger("calibration")

BASELINE_HAND_SPAN = 0.20  # Standard normalized distance between wrist and middle MCP


class CalibrationDialog(QDialog):
    """Interactive wizard measuring hand span and auto-tuning scale factors."""

    calibration_completed = Signal(float)  # Emits new calibration_scale

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("AirControl - Hand Size Calibration")
        self.setFixedSize(480, 360)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._collected_spans: List[float] = []
        self._target_samples = 35
        self._is_calibrating = False

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title and instructions
        title = QLabel("Auto-Scale Hand Calibration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(title)

        desc = QLabel(
            "Position your hand naturally in front of your camera showing an "
            "<b>OPEN PALM</b> at your normal comfortable working distance.<br><br>"
            "AirControl will measure your hand span and auto-scale distance thresholds "
            "so cursor movements and pinch gestures fit your hand size perfectly."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #cbd5e1; font-size: 13px; line-height: 1.4;")
        layout.addWidget(desc)

        # Status group
        status_box = QGroupBox("Calibration Telemetry")
        status_box.setStyleSheet("QGroupBox { font-weight: bold; color: #94a3b8; }")
        status_layout = QVBoxLayout(status_box)

        self.status_label = QLabel("Status: Click 'Begin Calibration' to start.")
        self.status_label.setStyleSheet("color: #f8fafc; font-size: 13px;")
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self._target_samples)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                height: 16px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #38bdf8;
                border-radius: 5px;
            }
        """)
        status_layout.addWidget(self.progress_bar)

        self.result_label = QLabel(f"Current Calibration Scale: {self.config.preferences.calibration_scale:.2f}x")
        self.result_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        status_layout.addWidget(self.result_label)

        layout.addWidget(status_box)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.start_btn = QPushButton("Begin Calibration")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: white;
                font-weight: bold;
                padding: 8px 18px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        self.start_btn.clicked.connect(self._toggle_calibration)
        btn_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Done")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: white;
                padding: 8px 18px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        self.cancel_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def _toggle_calibration(self) -> None:
        if self._is_calibrating:
            self._is_calibrating = False
            self.start_btn.setText("Begin Calibration")
            self.status_label.setText("Status: Calibration paused.")
        else:
            self._is_calibrating = True
            self._collected_spans.clear()
            self.progress_bar.setValue(0)
            self.start_btn.setText("Cancel Sampling")
            self.status_label.setText("Status: Hold open palm steady towards camera...")

    def feed_frame_data(self, hand: Optional[HandData], classification: ClassificationResult) -> None:
        """Receive frame data from main tracking loop while dialog is open."""
        if not self._is_calibrating or hand is None:
            return

        if classification.gesture == StaticGesture.OPEN_PALM:
            self._collected_spans.append(hand.hand_span)
            count = len(self._collected_spans)
            self.progress_bar.setValue(count)
            self.status_label.setText(f"Sampling palm geometry: {count}/{self._target_samples} frames...")

            if count >= self._target_samples:
                # Finished sampling!
                self._is_calibrating = False
                self.start_btn.setText("Re-Calibrate")

                # Compute robust trimmed mean
                spans = sorted(self._collected_spans)
                # Trim 10% outliers from ends
                trim = max(1, int(len(spans) * 0.10))
                valid_spans = spans[trim:-trim] if len(spans) > 2 * trim else spans
                mean_span = float(np.mean(valid_spans))

                scale = mean_span / BASELINE_HAND_SPAN
                scale = max(0.5, min(2.5, scale))

                self.config.preferences.calibration_scale = scale
                self.config.save()

                self.status_label.setText("Calibration complete! Scale saved.")
                self.result_label.setText(
                    f"Measured hand span: {mean_span:.3f} (Scale: {scale:.2f}x applied)"
                )
                self.calibration_completed.emit(scale)
                logger.info("Hand calibration completed: span=%.3f, scale=%.2f", mean_span, scale)
        else:
            self.status_label.setText("Please open your palm fully to continue sampling.")

