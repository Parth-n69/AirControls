"""Windows system tray icon manager for AirControl.

Enables background operation, tray notifications, and quick control toggling.
"""

from __future__ import annotations
from typing import Optional, Callable
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import QSystemTrayIcon, QMenu

from aircontrol.utils.logger import get_logger

logger = get_logger("tray")


def create_tray_pixmap(is_active: bool = False) -> QPixmap:
    """Generate a clean 32x32 dynamic tray icon."""
    pix = QPixmap(32, 32)
    pix.fill(Qt.transparent)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)

    # Background circle
    bg_color = QColor("#0284c7") if is_active else QColor("#475569")
    painter.setBrush(bg_color)
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(2, 2, 28, 28)

    # Inner letter "A"
    painter.setPen(QColor("#ffffff"))
    font = QFont("Arial", 14, QFont.Bold)
    painter.setFont(font)
    painter.drawText(pix.rect(), Qt.AlignCenter, "A")
    painter.end()

    return pix


class AirControlTrayIcon(QSystemTrayIcon):
    """System tray icon with menu actions and background integration."""

    def __init__(
        self,
        parent=None,
        on_show: Optional[Callable[[], None]] = None,
        on_toggle: Optional[Callable[[], None]] = None,
        on_exit: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.on_show = on_show
        self.on_toggle = on_toggle
        self.on_exit = on_exit

        self.update_state_icon(is_active=False)
        self.setToolTip("AirControl - Hand Gesture Desktop Control")

        self._build_menu()
        self.activated.connect(self._on_tray_activated)

    def _build_menu(self) -> None:
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu { background-color: #1e293b; color: white; border: 1px solid #334155; }
            QMenu::item:selected { background-color: #0284c7; }
        """)

        show_action = menu.addAction("Show AirControl")
        if self.on_show:
            show_action.triggered.connect(self.on_show)

        self.toggle_action = menu.addAction("Toggle Control (ON/OFF)")
        if self.on_toggle:
            self.toggle_action.triggered.connect(self.on_toggle)

        menu.addSeparator()

        exit_action = menu.addAction("Exit AirControl")
        if self.on_exit:
            exit_action.triggered.connect(self.on_exit)

        self.setContextMenu(menu)

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            if self.on_show:
                self.on_show()

    def update_state_icon(self, is_active: bool) -> None:
        """Update icon color based on control state."""
        pixmap = create_tray_pixmap(is_active)
        self.setIcon(QIcon(pixmap))

    def show_notification(self, title: str, message: str) -> None:
        """Display a balloon / toast notification from tray."""
        self.showMessage(title, message, QSystemTrayIcon.Information, 2000)

