"""Pose-normalized geometric facial features (inputs to the AU regression core).

All distances are in inter-ocular-distance (IOD) units in the face frame
(y down), so values are comparable across subjects and camera distances.
"""

from __future__ import annotations

import numpy as np

from cogsense import landmarks as L

FEATURE_NAMES = (
    "brow_inner_h_r", "brow_inner_h_l",
    "brow_outer_h_r", "brow_outer_h_l",
    "brow_gap",
    "ear_r", "ear_l",
    "lower_lid_r", "lower_lid_l",
    "lip_corner_depth", "mouth_width",
    "mouth_aperture", "jaw_drop", "lip_thickness",
)


def _mean(p: np.ndarray, idx) -> np.ndarray:
    return p[list(idx)].mean(axis=0)


def _ear(p: np.ndarray, idx) -> float:
    p1, p2, p3, p4, p5, p6 = (p[i] for i in idx)
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-9:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horiz))


def extract(p: np.ndarray) -> dict[str, float]:
    """Compute the feature vector from normalized face-frame points."""
    r_corner_y = (p[L.R_EYE_OUTER, 1] + p[L.R_EYE_INNER, 1]) / 2
    l_corner_y = (p[L.L_EYE_OUTER, 1] + p[L.L_EYE_INNER, 1]) / 2
    r_inner = _mean(p, L.R_BROW_INNER)
    l_inner = _mean(p, L.L_BROW_INNER)
    r_outer = _mean(p, L.R_BROW_OUTER)
    l_outer = _mean(p, L.L_BROW_OUTER)

    lip_center = (p[L.LIP_UPPER_OUTER] + p[L.LIP_LOWER_OUTER]) / 2
    corners = (p[L.MOUTH_R] + p[L.MOUTH_L]) / 2

    return {
        # Brow height above the eye-corner line (larger = raised).
        "brow_inner_h_r": float(r_corner_y - r_inner[1]),
        "brow_inner_h_l": float(l_corner_y - l_inner[1]),
        "brow_outer_h_r": float(p[L.R_EYE_OUTER, 1] - r_outer[1]),
        "brow_outer_h_l": float(p[L.L_EYE_OUTER, 1] - l_outer[1]),
        "brow_gap": float(abs(l_inner[0] - r_inner[0])),
        "ear_r": _ear(p, L.R_EYE_EAR),
        "ear_l": _ear(p, L.L_EYE_EAR),
        # Lower lid position below the corner line (smaller = lid raised, AU7).
        "lower_lid_r": float(_mean(p, L.R_EYE_LOWER)[1] - r_corner_y),
        "lower_lid_l": float(_mean(p, L.L_EYE_LOWER)[1] - l_corner_y),
        "lip_corner_depth": float(corners[2] - lip_center[2]),
        "mouth_width": float(np.linalg.norm(p[L.MOUTH_L] - p[L.MOUTH_R])),
        "mouth_aperture": float(np.linalg.norm(p[L.LIP_LOWER_INNER] - p[L.LIP_UPPER_INNER])),
        "jaw_drop": float(np.linalg.norm(p[L.CHIN] - p[L.NOSE_TIP])),
        # Visible (vermilion) thickness of both lips; pressing the lips together thins it (AU24).
        "lip_thickness": float(np.linalg.norm(p[L.LIP_UPPER_OUTER] - p[L.LIP_UPPER_INNER])
                               + np.linalg.norm(p[L.LIP_LOWER_OUTER] - p[L.LIP_LOWER_INNER])),
    }


def vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features[k] for k in FEATURE_NAMES], dtype=np.float64)
