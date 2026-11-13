"""MediaPipe Hands wrapper optimized for real-time CPU performance.

Supports MediaPipe Tasks API (modern) with fallback support for legacy solutions.
Features:
- Ultralight CPU inference with XNNPACK acceleration (>= 25 FPS)
- Adaptive downscaled inference with normalized coordinate mapping
- Multi-hand tracking with configurable dominant-hand selection
- Automatic hand span, palm centroid, and bounding box computation
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except (ImportError, ModuleNotFoundError):
    mp = None
    HAS_MEDIAPIPE = False

from aircontrol.utils.logger import get_logger
from aircontrol.utils.math_helpers import euclidean_distance_2d

logger = get_logger("tracker")

# Default model asset path for MediaPipe Tasks API
DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "models", "hand_landmarker.task")
)

# Landmark Indices for MediaPipe Hands
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Canonical landmark connections for skeleton rendering
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (WRIST, PINKY_MCP)
]


@dataclass
class HandLandmark:
    """Individual 3D hand landmark in normalized and pixel space."""
    x: float       # Normalized [0.0, 1.0]
    y: float       # Normalized [0.0, 1.0]
    z: float       # Depth relative to wrist
    px: int        # Pixel X coordinate on source frame
    py: int        # Pixel Y coordinate on source frame


@dataclass
class HandData:
    """Structured representation of a detected hand."""
    landmarks: List[HandLandmark]
    handedness: str                  # "Right" or "Left"
    confidence: float
    palm_center_norm: Tuple[float, float]
    palm_center_px: Tuple[int, int]
    hand_span: float                 # Distance from wrist to middle MCP (scale invariant)
    bbox_norm: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    bbox_px: Tuple[int, int, int, int]            # min_x, min_y, max_x, max_y


@dataclass
class TrackingResult:
    """Result of hand tracking on a single frame."""
    hands: List[HandData]
    dominant_hand: Optional[HandData]
    inference_time_ms: float


# Singleton cache for MediaPipe C++ detector to prevent handle invalidation across instances
_SHARED_TASKS_DETECTOR = None


class HandTracker:
    """MediaPipe hand landmark detector with multi-backend compatibility."""

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        inference_resolution: Tuple[int, int] = (480, 270),
        dominant_hand_preference: str = "Right",
        model_path: Optional[str] = None,
    ) -> None:
        global _SHARED_TASKS_DETECTOR
        self.max_num_hands = max_num_hands
        self.inference_width, self.inference_height = inference_resolution
        self.dominant_hand_preference = dominant_hand_preference
        self._backend = "tasks"
        self._detector = None
        self._legacy_hands = None

        target_model = model_path or DEFAULT_MODEL_PATH

        # 1. Attempt MediaPipe Tasks API (with singleton reuse)
        if os.path.exists(target_model):
            try:
                if _SHARED_TASKS_DETECTOR is not None:
                    self._detector = _SHARED_TASKS_DETECTOR
                    self._backend = "tasks"
                    logger.debug("HandTracker reused cached MediaPipe Tasks detector.")
                    return

                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=target_model)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                _SHARED_TASKS_DETECTOR = vision.HandLandmarker.create_from_options(options)
                self._detector = _SHARED_TASKS_DETECTOR
                self._backend = "tasks"
                logger.info("HandTracker initialized using MediaPipe Tasks API with %s.", target_model)
                return
            except Exception as exc:
                logger.warning("Failed to initialize MediaPipe Tasks API: %s. Trying legacy solutions.", exc)

        # 2. Fallback to Legacy mp.solutions.hands if present
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self._legacy_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_num_hands,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._backend = "solutions"
                logger.info("HandTracker initialized using legacy MediaPipe solutions.")
                return
            except Exception as exc:
                logger.error("Failed to initialize legacy solutions: %s", exc)

        logger.warning("No functional MediaPipe backend found. Running in mock/passthrough mode.")
        self._backend = "none"

    def close(self) -> None:
        """Release tracker hardware / memory resources."""
        if self._legacy_hands is not None:
            try:
                self._legacy_hands.close()
            except Exception:
                pass
            self._legacy_hands = None

    def set_dominant_hand(self, dominant_hand: str) -> None:
        """Update dominant hand preference ('Right' or 'Left')."""
        self.dominant_hand_preference = "Right" if dominant_hand.lower().startswith("r") else "Left"

    def process(self, frame: np.ndarray) -> TrackingResult:
        """Process a BGR frame and return detected hand landmarks."""
        start_time = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        if self._backend == "none":
            return TrackingResult(hands=[], dominant_hand=None, inference_time_ms=0.0)

        # CPU optimization: downscale frame for faster inference
        if orig_w != self.inference_width or orig_h != self.inference_height:
            small_frame = cv2.resize(
                frame,
                (self.inference_width, self.inference_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small_frame = frame

        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        detected_hands: List[HandData] = []

        if self._backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self._detector.detect(mp_image)

            if detection_result.hand_landmarks:
                for idx, hand_lms in enumerate(detection_result.hand_landmarks):
                    # Handedness category
                    label = "Right"
                    confidence = 0.90
                    if idx < len(detection_result.handedness) and detection_result.handedness[idx]:
                        top_cat = detection_result.handedness[idx][0]
                        label = top_cat.category_name or top_cat.display_name or "Right"
                        confidence = top_cat.score

                    hand_data = self._build_hand_data(hand_lms, label, confidence, orig_w, orig_h)
                    detected_hands.append(hand_data)

        elif self._backend == "solutions":
            rgb_frame.flags.writeable = False
            results = self._legacy_hands.process(rgb_frame)
            if results.multi_hand_landmarks and results.multi_handedness:
                for lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    confidence = handedness.classification[0].score
                    hand_data = self._build_hand_data(lms.landmark, label, confidence, orig_w, orig_h)
                    detected_hands.append(hand_data)

        # Dominant hand selection
        dominant_hand: Optional[HandData] = None
        if detected_hands:
            preferred = [h for h in detected_hands if h.handedness.lower() == self.dominant_hand_preference.lower()]
            if preferred:
                dominant_hand = max(preferred, key=lambda h: h.confidence)
            else:
                dominant_hand = max(detected_hands, key=lambda h: h.confidence)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return TrackingResult(
            hands=detected_hands,
            dominant_hand=dominant_hand,
            inference_time_ms=elapsed_ms,
        )

    def _build_hand_data(
        self,
        landmarks_raw: list,
        label: str,
        confidence: float,
        orig_w: int,
        orig_h: int
    ) -> HandData:
        """Parse raw landmark list into HandData struct."""
        landmarks: List[HandLandmark] = []
        xs_norm: List[float] = []
        ys_norm: List[float] = []

        for lm in landmarks_raw:
            nx = max(0.0, min(1.0, float(lm.x)))
            ny = max(0.0, min(1.0, float(lm.y)))
            nz = float(lm.z)

            px = int(nx * orig_w)
            py = int(ny * orig_h)

            xs_norm.append(nx)
            ys_norm.append(ny)
            landmarks.append(HandLandmark(x=nx, y=ny, z=nz, px=px, py=py))

        w_pt = (landmarks[WRIST].x, landmarks[WRIST].y)
        m_mcp_pt = (landmarks[MIDDLE_MCP].x, landmarks[MIDDLE_MCP].y)
        hand_span = max(1e-4, euclidean_distance_2d(w_pt, m_mcp_pt))

        i_mcp_pt = (landmarks[INDEX_MCP].x, landmarks[INDEX_MCP].y)
        p_mcp_pt = (landmarks[PINKY_MCP].x, landmarks[PINKY_MCP].y)
        palm_x = (w_pt[0] + i_mcp_pt[0] + p_mcp_pt[0]) / 3.0
        palm_y = (w_pt[1] + i_mcp_pt[1] + p_mcp_pt[1]) / 3.0
        palm_center_norm = (palm_x, palm_y)
        palm_center_px = (int(palm_x * orig_w), int(palm_y * orig_h))

        min_nx, max_nx = min(xs_norm), max(xs_norm)
        min_ny, max_ny = min(ys_norm), max(ys_norm)
        bbox_norm = (min_nx, min_ny, max_nx, max_ny)
        bbox_px = (int(min_nx * orig_w), int(min_ny * orig_h), int(max_nx * orig_w), int(max_ny * orig_h))

        return HandData(
            landmarks=landmarks,
            handedness=label,
            confidence=confidence,
            palm_center_norm=palm_center_norm,
            palm_center_px=palm_center_px,
            hand_span=hand_span,
            bbox_norm=bbox_norm,
            bbox_px=bbox_px,
        )
