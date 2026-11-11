"""PyInstaller packaging script for AirControl on Windows.

Builds a standalone, zero-dependency executable bundling:
- PySide6 runtime and Qt plugins
- MediaPipe models (hand_landmarker.task)
- PyAutoGUI, pycaw, and OpenCV dependencies
- Default configuration and asset files
"""

from __future__ import annotations
import os
import sys
import shutil
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENTRY_POINT = os.path.join(PROJECT_DIR, "aircontrol", "app.py")
OUTPUT_NAME = "AirControl"
MODEL_SRC = os.path.join(PROJECT_DIR, "aircontrol", "assets", "models", "hand_landmarker.task")
MODEL_DEST = os.path.join("aircontrol", "assets", "models")


def build() -> int:
    """Execute PyInstaller build pipeline."""
    print("=" * 60)
    print("Packaging AirControl for Windows via PyInstaller...")
    print("=" * 60)

    # Ensure model file is present
    if not os.path.exists(MODEL_SRC):
        print(f"Error: Required model file not found at: {MODEL_SRC}")
        return 1

    # Clean previous build artifacts
    for dir_name in ["build", "dist"]:
        path = os.path.join(PROJECT_DIR, dir_name)
        if os.path.exists(path):
            print(f"Cleaning {path}...")
            shutil.rmtree(path, ignore_errors=True)

    # Data files to include: source;destination_folder
    # On Windows, PyInstaller separator is ';'
    add_data_args = [
        f"--add-data={MODEL_SRC};{MODEL_DEST}",
        f"--add-data={os.path.join(PROJECT_DIR, 'config.json')};.",
    ]

    # Hidden imports required by dynamic plugin loaders
    hidden_imports = [
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=cv2",
        "--hidden-import=mediapipe",
        "--hidden-import=pycaw",
        "--hidden-import=comtypes",
        "--hidden-import=pynput",
        "--hidden-import=pyautogui",
    ]

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=" + OUTPUT_NAME,
        "--onedir",             # Production onedir for fast startup
        "--windowed",           # Hide console in production
        "--clean",
        *add_data_args,
        *hidden_imports,
        ENTRY_POINT,
    ]

    print("Running command:")
    print(" ".join(cmd))
    print("-" * 60)

    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode == 0:
        dist_path = os.path.join(PROJECT_DIR, "dist", OUTPUT_NAME)
        print("=" * 60)
        print(f"Build succeeded! Standalone executable directory created at:\n{dist_path}")
        print(f"Main Executable: {os.path.join(dist_path, OUTPUT_NAME + '.exe')}")
        print("=" * 60)
    else:
        print(f"Build failed with exit code {result.returncode}.")

    return result.returncode


if __name__ == "__main__":
    sys.exit(build())

