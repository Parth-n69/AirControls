"""Scale-invariant static hand pose classifier for AirControl.

Classifies hand landmarks into distinct canonical poses:
- FIST: All fingers curled into palm (used for toggling system control ON/OFF)
- POINT: Index finger extended, others curled (cursor navigation)
- PINCH: Thumb tip and index tip in close proximity (mouse click / drag)
- TWO_FINGERS_UP: Index and middle fingers extended (page scrolling)
- THREE_FINGERS_UP: Index, middle, and ring fingers extended (continuous volume)
- OPEN_PALM: All five fingers extended (base pose for directional swipes)
- UNKNOWN: Transitional or unclassified pose

All distance thresholds are normalized by the hand span (Wrist to Middle MCP)
and the user's calibrated hand scale, ensuring invariant recognition across
different hand sizes and camera distances.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

from aircontrol.core.tracker import (
    HandData,
    WRIST,
    THUMB_CMC,
    THUMB_MCP,
    THUMB_IP,
    THUMB_TIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_DIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_DIP,
    MIDDLE_TIP,
    RING_MCP,
    RING_PIP,
    RING_DIP,
    RING_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_DIP,
    PINKY_TIP,
)
from aircontrol.utils.math_helpers import euclidean_distance_2d
from aircontrol.core.config import GestureThresholds


class StaticGesture(str, Enum):
    """Enumeration of recognized static hand poses."""
    NONE = "NONE"
    FIST = "FIST"
    POINT = "POINT"
    PINCH = "PINCH"
    TWO_FINGERS_UP = "TWO_FINGERS_UP"
    THREE_FINGERS_UP = "THREE_FINGERS_UP"
    OPEN_PALM = "OPEN_PALM"
    UNKNOWN = "UNKNOWN"


@dataclass
class FingerState:
    """Extension state for the five hand digits."""
    thumb_extended: bool
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool


@dataclass
class ClassificationResult:
    """Output of static pose classification."""
    gesture: StaticGesture
    confidence: float
    finger_state: FingerState
    pinch_distance_ratio: float
    normalized_hand_span: float


class GestureClassifier:
    """Classifies hand poses using scale-normalized geometric heuristics."""

    def __init__(self, thresholds: GestureThresholds, calibration_scale: float = 1.0) -> None:
        self.thresholds = thresholds
        self.calibration_scale = max(0.1, calibration_scale)
        self._is_pinching = False  # Hysteresis state

    def update_calibration_scale(self, scale: float) -> None:
        """Update calibration scaling factor."""
        self.calibration_scale = max(0.1, scale)

    def classify(self, hand: HandData) -> ClassificationResult:
        """Classify hand landmarks into a static gesture.

        Args:
            hand: Detected HandData with 21 landmarks.

        Returns:
            ClassificationResult containing the classified gesture and telemetry.
        """
        lms = hand.landmarks
        # Reference scale: distance between wrist (0) and middle MCP (9)
        ref_span = max(1e-4, hand.hand_span * self.calibration_scale)

        # 1. Compute Tip-to-Wrist and Tip-to-MCP distances normalized by hand span
        wrist_pt = (lms[WRIST].x, lms[WRIST].y)

        # Finger tip and MCP points
        thumb_tip = (lms[THUMB_TIP].x, lms[THUMB_TIP].y)
        thumb_ip = (lms[THUMB_IP].x, lms[THUMB_IP].y)
        thumb_mcp = (lms[THUMB_MCP].x, lms[THUMB_MCP].y)

        index_tip = (lms[INDEX_TIP].x, lms[INDEX_TIP].y)
        index_pip = (lms[INDEX_PIP].x, lms[INDEX_PIP].y)
        index_mcp = (lms[INDEX_MCP].x, lms[INDEX_MCP].y)

        middle_tip = (lms[MIDDLE_TIP].x, lms[MIDDLE_TIP].y)
        middle_pip = (lms[MIDDLE_PIP].x, lms[MIDDLE_PIP].y)
        middle_mcp = (lms[MIDDLE_MCP].x, lms[MIDDLE_MCP].y)

        ring_tip = (lms[RING_TIP].x, lms[RING_TIP].y)
        ring_pip = (lms[RING_PIP].x, lms[RING_PIP].y)
        ring_mcp = (lms[RING_MCP].x, lms[RING_MCP].y)

        pinky_tip = (lms[PINKY_TIP].x, lms[PINKY_TIP].y)
        pinky_pip = (lms[PINKY_PIP].x, lms[PINKY_PIP].y)
        pinky_mcp = (lms[PINKY_MCP].x, lms[PINKY_MCP].y)

        # 2. Determine extension for fingers 1..4 (Index, Middle, Ring, Pinky)
        # A finger is extended if dist(tip, wrist) > dist(pip, wrist) + buffer
        # AND dist(tip, mcp) > dist(pip, mcp)
        index_ext = (
            euclidean_distance_2d(index_tip, wrist_pt) > euclidean_distance_2d(index_pip, wrist_pt) + 0.15 * ref_span
            and euclidean_distance_2d(index_tip, index_mcp) > euclidean_distance_2d(index_pip, index_mcp)
        )
        middle_ext = (
            euclidean_distance_2d(middle_tip, wrist_pt) > euclidean_distance_2d(middle_pip, wrist_pt) + 0.15 * ref_span
            and euclidean_distance_2d(middle_tip, middle_mcp) > euclidean_distance_2d(middle_pip, middle_mcp)
        )
        ring_ext = (
            euclidean_distance_2d(ring_tip, wrist_pt) > euclidean_distance_2d(ring_pip, wrist_pt) + 0.15 * ref_span
            and euclidean_distance_2d(ring_tip, ring_mcp) > euclidean_distance_2d(ring_pip, ring_mcp)
        )
        pinky_ext = (
            euclidean_distance_2d(pinky_tip, wrist_pt) > euclidean_distance_2d(pinky_pip, wrist_pt) + 0.15 * ref_span
            and euclidean_distance_2d(pinky_tip, pinky_mcp) > euclidean_distance_2d(pinky_pip, pinky_mcp)
        )

        # Thumb extension: tip distance from index MCP and thumb MCP
        thumb_ext = (
            euclidean_distance_2d(thumb_tip, index_mcp) > 0.32 * ref_span
            and euclidean_distance_2d(thumb_tip, wrist_pt) > euclidean_distance_2d(thumb_ip, wrist_pt)
        )

        finger_state = FingerState(
            thumb_extended=thumb_ext,
            index_extended=index_ext,
            middle_extended=middle_ext,
            ring_extended=ring_ext,
            pinky_extended=pinky_ext,
        )

        # 3. Calculate pinch distance ratio: dist(thumb_tip, index_tip) / ref_span
        raw_pinch_dist = euclidean_distance_2d(thumb_tip, index_tip)
        pinch_ratio = raw_pinch_dist / ref_span

        # Pinch detection with hysteresis:
        # Enter pinch at pinch_distance_ratio (e.g. 0.35 * ref_span)
        # Release pinch at pinch_release_ratio (e.g. 0.48 * ref_span)
        pinch_enter_thresh = self.thresholds.pinch_distance_ratio * 5.0
        pinch_release_thresh = self.thresholds.pinch_release_ratio * 5.0

        if self._is_pinching:
            if pinch_ratio > pinch_release_thresh:
                self._is_pinching = False
        else:
            if pinch_ratio < pinch_enter_thresh:
                # Also ensure middle/ring/pinky are not all wide open
                if not (middle_ext and ring_ext and pinky_ext):
                    self._is_pinching = True

        # 4. Pose Decision Tree
        classified_gesture = StaticGesture.UNKNOWN
        confidence = 0.85

        # Fist: all 4 main fingers curled tight, thumb curled
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
            classified_gesture = StaticGesture.FIST
            confidence = 0.95

        # Pinch takes priority when pinching is actively engaged
        elif self._is_pinching:
            classified_gesture = StaticGesture.PINCH
            confidence = 0.90

        # Open Palm: all 4 main fingers extended and not pinching
        elif index_ext and middle_ext and ring_ext and pinky_ext and not self._is_pinching:
            classified_gesture = StaticGesture.OPEN_PALM
            confidence = 0.95

        # Three Fingers Up: Index, Middle, Ring extended, Pinky curled
        elif index_ext and middle_ext and ring_ext and not pinky_ext:
            classified_gesture = StaticGesture.THREE_FINGERS_UP
            confidence = 0.90

        # Two Fingers Up: Index and Middle extended, Ring and Pinky curled
        elif index_ext and middle_ext and not ring_ext and not pinky_ext:
            classified_gesture = StaticGesture.TWO_FINGERS_UP
            confidence = 0.92

        # Point: Index extended, Middle, Ring, Pinky curled
        elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
            classified_gesture = StaticGesture.POINT
            confidence = 0.94

        else:
            classified_gesture = StaticGesture.UNKNOWN
            confidence = 0.50

        return ClassificationResult(
            gesture=classified_gesture,
            confidence=confidence,
            finger_state=finger_state,
            pinch_distance_ratio=pinch_ratio,
            normalized_hand_span=ref_span,
        )
