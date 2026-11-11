"""Bootstrap entry point for the AirControl desktop application."""

from __future__ import annotations
import sys
import os
import argparse
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from aircontrol.core.config import AppConfig
from aircontrol.utils.logger import setup_logger, get_logger
from aircontrol.ui.main_window import MainWindow

logger = get_logger("app")


def main() -> int:
    """Initialize application, parse arguments, and run Qt event loop."""
    parser = argparse.ArgumentParser(description="AirControl: Mid-air hand gesture desktop control")
    parser.add_argument("--tray", action="store_true", help="Start minimized in system tray")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json file")
    args = parser.parse_args()

    # 1. Initialize logging (writes to aircontrol.log, no frames ever recorded)
    setup_logger()
    logger.info("Starting AirControl desktop application...")

    # 2. Configure Qt Application
    # High-DPI attributes are enabled by default in Qt 6
    app = QApplication(sys.argv)
    app.setApplicationName("AirControl")
    app.setOrganizationName("AirControl")
    app.setQuitOnLastWindowClosed(False)

    # 3. Load Configuration
    config = AppConfig.load(args.config)

    # 4. Create Main Window
    window = MainWindow(config)

    start_in_tray = args.tray or config.preferences.start_in_tray
    if start_in_tray:
        logger.info("Starting AirControl minimized to system tray.")
        window.hide()
        window.tray_icon.show_notification("AirControl", "AirControl started in background tray.")
    else:
        window.show()
        window.raise_()
        window.activateWindow()

    # 5. Run Application Event Loop
    exit_code = app.exec()
    logger.info("AirControl terminated with exit code %d.", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

