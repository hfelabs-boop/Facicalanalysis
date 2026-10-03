"""Head-pose estimation and rigid normalization of face landmarks (FR-2.1).

Pose is computed from a face-anchored coordinate frame built from rigid
landmarks (eye corners, face sides, forehead, chin). Landmarks are then
rotated into that frame and scaled by inter-ocular distance (IOD), so the
AU features downstream are invariant to head rotation, translation and
camera distance.

Conventions (camera frame): x → image right, y → image down, z → away
from camera. ``yaw > 0`` turns the nose toward the image left, ``pitch > 0``
raises the chin, ``roll > 0`` rotates the face clockwise in the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cogsense import landmarks as L


def rotation_from_euler(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R = Ry(yaw) @ Rx(pitch) @ Rz(roll); maps face-frame vectors into the camera frame."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return ry @ rx @ rz


def euler_from_rotation(rot: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rotation_from_euler`; returns (yaw, pitch, roll) in degrees."""
    pitch = np.arcsin(np.clip(-rot[1, 2], -1.0, 1.0))
    yaw = np.arctan2(rot[0, 2], rot[2, 2])
    roll = np.arctan2(rot[1, 0], rot[1, 1])
    return tuple(float(v) for v in np.degrees([yaw, pitch, roll]))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


@dataclass
class NormalizedFace:
    points: np.ndarray  # (N, 3) in face frame, IOD units, origin at nose bridge
    rotation: np.ndarray  # (3, 3) face→camera
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    iod_px: float


def to_pixel_space(landmarks_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """MediaPipe normalized (x/W, y/H, z/W) → isotropic pixel coordinates."""
    pts = np.asarray(landmarks_norm, dtype=np.float64)
    return pts * np.array([width, height, width], dtype=np.float64)


def face_frame(points_px: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (R face→camera, IOD in px) from rigid landmarks."""
    p = points_px
    r_eye = (p[L.R_EYE_OUTER] + p[L.R_EYE_INNER]) / 2
    l_eye = (p[L.L_EYE_OUTER] + p[L.L_EYE_INNER]) / 2
    iod = float(np.linalg.norm(l_eye - r_eye))
    ex = _unit(
        (p[L.L_EYE_OUTER] - p[L.R_EYE_OUTER])
        + (p[L.L_EYE_INNER] - p[L.R_EYE_INNER])
        + 0.5 * (p[L.FACE_L] - p[L.FACE_R])
    )
    down = p[L.CHIN] - p[L.FOREHEAD]
    ey = _unit(down - np.dot(down, ex) * ex)
    ez = np.cross(ex, ey)
    return np.column_stack([ex, ey, ez]), iod


def normalize(points_px: np.ndarray) -> NormalizedFace:
    rot, iod = face_frame(points_px)
    origin = points_px[L.NOSE_BRIDGE]
    pts = (points_px - origin) @ rot / max(iod, 1e-6)
    yaw, pitch, roll = euler_from_rotation(rot)
    return NormalizedFace(pts, rot, yaw, pitch, roll, iod)
