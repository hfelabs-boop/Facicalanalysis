"""Parametric synthetic face mesh for tests, demos and the scenario simulator.

Produces MediaPipe-style normalized landmarks (478 × 3) for a given set of
AU activations and head pose. Only the landmarks used by the AU core are
anatomically placed; the rest lie on a deterministic ellipsoid so the
array shape matches the real tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cogsense import landmarks as L
from cogsense.geometry import rotation_from_euler


def _mirror(points: dict[int, tuple[float, float, float]], pairs: list[tuple[int, int]]) -> None:
    for right, left in pairs:
        x, y, z = points[right]
        points[left] = (-x, y, z)


def _template() -> np.ndarray:
    rng = np.random.default_rng(7)
    n = L.NUM_LANDMARKS
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(-1.2, 1.2, n)
    base = np.column_stack([1.1 * np.cos(phi) * np.sin(theta), 0.4 + 1.3 * np.sin(phi), 0.6 - 0.6 * np.cos(phi) * np.cos(theta)])

    p: dict[int, tuple[float, float, float]] = {
        L.NOSE_BRIDGE: (0.0, 0.0, 0.0),
        6: (0.0, 0.25, -0.05),
        L.NOSE_TIP: (0.0, 0.85, -0.45),
        L.FOREHEAD: (0.0, -0.95, 0.05),
        L.CHIN: (0.0, 1.85, 0.05),
        L.FACE_R: (-1.25, 0.4, 0.9),
        # right eye
        33: (-0.78, 0.17, 0.20), 133: (-0.24, 0.17, 0.12),
        160: (-0.60, 0.08, 0.10), 159: (-0.51, 0.06, 0.08), 158: (-0.42, 0.07, 0.08),
        144: (-0.60, 0.24, 0.12), 145: (-0.51, 0.25, 0.10), 153: (-0.42, 0.24, 0.10),
        # right brow
        107: (-0.20, -0.22, 0.02), 55: (-0.22, -0.17, 0.03),
        105: (-0.48, -0.30, 0.05), 66: (-0.38, -0.27, 0.04), 65: (-0.45, -0.22, 0.05),
        70: (-0.85, -0.18, 0.25), 46: (-0.83, -0.12, 0.25),
        # mouth
        L.MOUTH_R: (-0.42, 1.25, 0.05),
        L.LIP_UPPER_OUTER: (0.0, 1.08, -0.12), L.LIP_LOWER_OUTER: (0.0, 1.42, -0.08),
        L.LIP_UPPER_INNER: (0.0, 1.20, -0.08), L.LIP_LOWER_INNER: (0.0, 1.24, -0.08),
    }
    _mirror(p, [(L.FACE_R, L.FACE_L), (33, 263), (133, 362), (160, 387), (159, 386), (158, 385),
                (144, 373), (145, 374), (153, 380), (107, 336), (55, 285), (105, 334), (66, 296),
                (65, 295), (70, 300), (46, 276), (L.MOUTH_R, L.MOUTH_L)])
    for idx, xyz in p.items():
        base[idx] = xyz
    return base


_TEMPLATE = _template()


@dataclass
class FaceState:
    au1: float = 0.0
    au2: float = 0.0
    au4: float = 0.0
    au7: float = 0.0
    au14: float = 0.0
    blink: float = 0.0  # 0 open … 1 fully closed
    mouth_open: float = 0.0  # jaw drop for speech/chewing, 0…1
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    asym: dict = field(default_factory=dict)  # per-subject morphology offsets {idx: (dx,dy,dz)}


def deform(state: FaceState) -> np.ndarray:
    """Face-frame landmarks (IOD units) for the given activation state."""
    p = _TEMPLATE.copy()
    for idx, d in state.asym.items():
        p[idx] += d

    for side in (-1, 1):  # -1 subject right (image left), +1 subject left
        inner = L.R_BROW_INNER if side < 0 else L.L_BROW_INNER
        mid = L.R_BROW_MID if side < 0 else L.L_BROW_MID
        outer = L.R_BROW_OUTER if side < 0 else L.L_BROW_OUTER
        upper = L.R_EYE_UPPER if side < 0 else L.L_EYE_UPPER
        lower = L.R_EYE_LOWER if side < 0 else L.L_EYE_LOWER
        for i in inner:
            p[i, 1] += 0.08 * state.au4 - 0.09 * state.au1
            p[i, 0] -= side * 0.05 * state.au4
        for i in mid:
            p[i, 1] += 0.06 * state.au4 - 0.04 * state.au1 - 0.05 * state.au2
        for i in outer:
            p[i, 1] += 0.03 * state.au4 - 0.09 * state.au2
        for i in lower:
            p[i, 1] -= 0.05 * state.au7
        for u, lo in zip(upper, lower):
            p[u, 1] += 0.02 * state.au7
            p[u, 1] += state.blink * 0.92 * (p[lo, 1] - p[u, 1])
        corner = L.MOUTH_R if side < 0 else L.MOUTH_L
        p[corner, 2] += 0.06 * state.au14
        p[corner, 0] += side * 0.03 * state.au14

    jaw = 0.25 * state.mouth_open
    for i in (L.LIP_LOWER_INNER, L.LIP_LOWER_OUTER, L.CHIN):
        p[i, 1] += jaw
    p[L.LIP_UPPER_INNER, 1] -= 0.02 * state.mouth_open
    return p


def render(state: FaceState, width: int = 1920, height: int = 1080, iod_px: float = 120.0,
           center: tuple[float, float] | None = None, noise_px: float = 0.0,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """MediaPipe-normalized (x/W, y/H, z/W) landmarks for the state."""
    face = deform(state)
    rot = rotation_from_euler(state.yaw, state.pitch, state.roll)
    cx, cy = center if center else (width / 2, height / 2)
    px = (face @ rot.T) * iod_px + np.array([cx, cy, 0.0])
    if noise_px > 0:
        rng = rng or np.random.default_rng()
        px = px + rng.normal(0, noise_px, px.shape)
    return px / np.array([width, height, width])
