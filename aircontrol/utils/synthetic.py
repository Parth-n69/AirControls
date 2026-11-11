"""Synthetic hand generator for unit testing and demo simulation."""

from __future__ import annotations
from typing import Tuple, List
from aircontrol.core.tracker import HandData, HandLandmark


def create_synthetic_hand(
    wrist: Tuple[float, float] = (0.5, 0.8),
    hand_span: float = 0.20,
    finger_extensions: Tuple[bool, bool, bool, bool, bool] = (True, True, True, True, True),
    pinch: bool = False,
    scale: float = 1.0,
) -> HandData:
    """Generate a synthetic 21-landmark HandData for testing and demo simulation."""
    wx, wy = wrist
    eff_span = hand_span * scale
    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = finger_extensions

    lms: List[HandLandmark] = [HandLandmark(x=wx, y=wy, z=0.0, px=int(wx * 640), py=int(wy * 480))]

    # 1..4: Thumb
    t_dir_x = -0.4 * eff_span
    if pinch:
        lms.extend([
            HandLandmark(x=wx + t_dir_x * 0.3, y=wy - 0.2 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx + t_dir_x * 0.6, y=wy - 0.4 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx + t_dir_x * 0.8, y=wy - 0.6 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx - 0.05 * eff_span, y=wy - 1.2 * eff_span, z=0.0, px=0, py=0),
        ])
    elif thumb_ext:
        lms.extend([
            HandLandmark(x=wx + t_dir_x * 0.3, y=wy - 0.3 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx + t_dir_x * 0.6, y=wy - 0.5 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx + t_dir_x * 0.9, y=wy - 0.7 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx + t_dir_x * 1.3, y=wy - 0.9 * eff_span, z=0.0, px=0, py=0),
        ])
    else:
        lms.extend([
            HandLandmark(x=wx - 0.1 * eff_span, y=wy - 0.2 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx - 0.1 * eff_span, y=wy - 0.4 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx, y=wy - 0.4 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx, y=wy - 0.3 * eff_span, z=0.0, px=0, py=0),
        ])

    # 5..8: Index
    idx_mcp_x = wx - 0.15 * eff_span
    if pinch:
        lms.extend([
            HandLandmark(x=idx_mcp_x, y=wy - 0.8 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=idx_mcp_x, y=wy - 1.0 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=idx_mcp_x, y=wy - 1.1 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=wx - 0.05 * eff_span, y=wy - 1.2 * eff_span, z=0.0, px=0, py=0),
        ])
    elif index_ext:
        lms.extend([
            HandLandmark(x=idx_mcp_x, y=wy - 0.8 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=idx_mcp_x, y=wy - 1.1 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=idx_mcp_x, y=wy - 1.4 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=idx_mcp_x, y=wy - 1.7 * eff_span, z=0.0, px=0, py=0),
        ])
    else:
        lms.extend([
            HandLandmark(x=idx_mcp_x, y=wy - 0.8 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=idx_mcp_x, y=wy - 0.9 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=idx_mcp_x, y=wy - 0.7 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=idx_mcp_x, y=wy - 0.5 * eff_span, z=0.0, px=0, py=0),
        ])

    # 9..12: Middle
    mid_mcp_x = wx
    if middle_ext:
        lms.extend([
            HandLandmark(x=mid_mcp_x, y=wy - eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=mid_mcp_x, y=wy - 1.3 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=mid_mcp_x, y=wy - 1.6 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=mid_mcp_x, y=wy - 1.9 * eff_span, z=0.0, px=0, py=0),
        ])
    else:
        lms.extend([
            HandLandmark(x=mid_mcp_x, y=wy - eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=mid_mcp_x, y=wy - 1.1 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=mid_mcp_x, y=wy - 0.8 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=mid_mcp_x, y=wy - 0.6 * eff_span, z=0.0, px=0, py=0),
        ])

    # 13..16: Ring
    ring_mcp_x = wx + 0.15 * eff_span
    if ring_ext:
        lms.extend([
            HandLandmark(x=ring_mcp_x, y=wy - 0.8 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=ring_mcp_x, y=wy - 1.1 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=ring_mcp_x, y=wy - 1.4 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=ring_mcp_x, y=wy - 1.7 * eff_span, z=0.0, px=0, py=0),
        ])
    else:
        lms.extend([
            HandLandmark(x=ring_mcp_x, y=wy - 0.8 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=ring_mcp_x, y=wy - 0.9 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=ring_mcp_x, y=wy - 0.7 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=ring_mcp_x, y=wy - 0.5 * eff_span, z=0.0, px=0, py=0),
        ])

    # 17..20: Pinky
    pin_mcp_x = wx + 0.3 * eff_span
    if pinky_ext:
        lms.extend([
            HandLandmark(x=pin_mcp_x, y=wy - 0.6 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=pin_mcp_x, y=wy - 0.9 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=pin_mcp_x, y=wy - 1.1 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=pin_mcp_x, y=wy - 1.4 * eff_span, z=0.0, px=0, py=0),
        ])
    else:
        lms.extend([
            HandLandmark(x=pin_mcp_x, y=wy - 0.6 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=pin_mcp_x, y=wy - 0.7 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=pin_mcp_x, y=wy - 0.5 * eff_span, z=0.0, px=0, py=0),
            HandLandmark(x=pin_mcp_x, y=wy - 0.4 * eff_span, z=0.0, px=0, py=0),
        ])

    for lm in lms:
        lm.px = int(lm.x * 640)
        lm.py = int(lm.y * 480)

    return HandData(
        landmarks=lms,
        handedness="Right",
        confidence=0.95,
        palm_center_norm=(wx, wy - 0.5 * eff_span),
        palm_center_px=(int(wx * 640), int((wy - 0.5 * eff_span) * 480)),
        hand_span=eff_span,
        bbox_norm=(wx - 0.4 * eff_span, wy - 1.9 * eff_span, wx + 0.4 * eff_span, wy),
        bbox_px=(0, 0, 640, 480),
    )

