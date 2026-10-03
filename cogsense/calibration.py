"""15-second neutral baseline calibration (FR-1.3).

Collects pose-normalized features while the operator holds a relaxed,
neutral expression, then stores robust (median) resting values so AU
intensities are computed relative to the individual's own morphology —
resting brow height, natural asymmetry, habitual squint, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from cogsense.features import FEATURE_NAMES

BLENDSHAPE_KEYS = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthPressLeft", "mouthPressRight",
)


class CalibrationState(str, Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    COMPLETE = "COMPLETE"


@dataclass
class Baseline:
    features: dict[str, float]
    feature_mad: dict[str, float] = field(default_factory=dict)
    blendshapes: dict[str, float] = field(default_factory=dict)
    blink_rate_per_min: float | None = None
    frames: int = 0
    is_default: bool = False

    def to_dict(self) -> dict:
        return {
            "features": self.features, "feature_mad": self.feature_mad,
            "blendshapes": self.blendshapes, "blink_rate_per_min": self.blink_rate_per_min,
            "frames": self.frames, "is_default": self.is_default,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Baseline":
        return cls(d["features"], d.get("feature_mad", {}), d.get("blendshapes", {}),
                   d.get("blink_rate_per_min"), d.get("frames", 0), d.get("is_default", False))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        return cls.from_dict(json.loads(Path(path).read_text()))


class BaselineCalibrator:
    def __init__(self, duration_s: float = 15.0, min_frames: int = 90):
        self.duration_s = duration_s
        self.min_frames = min_frames
        self.state = CalibrationState.IDLE
        self._t0: float | None = None
        self._feats: list[np.ndarray] = []
        self._bs: list[dict[str, float]] = []
        self._blinks = 0

    def start(self, t: float) -> None:
        self.state = CalibrationState.COLLECTING
        self._t0 = t
        self._feats.clear()
        self._bs.clear()
        self._blinks = 0

    def progress(self, t: float) -> float:
        if self.state is CalibrationState.COMPLETE:
            return 1.0
        if self.state is not CalibrationState.COLLECTING or self._t0 is None:
            return 0.0
        return min((t - self._t0) / self.duration_s, 1.0)

    def add(self, t: float, feats: dict[str, float], blendshapes: dict[str, float] | None,
            eyes_closed: bool, blink_onset: bool, usable: bool) -> Baseline | None:
        """Feed one frame. Returns the Baseline once collection finishes."""
        if self.state is not CalibrationState.COLLECTING:
            return None
        if blink_onset:
            self._blinks += 1
        # Blinks, speech and low-confidence frames would bias the neutral reference.
        if usable and not eyes_closed:
            self._feats.append(np.array([feats[k] for k in FEATURE_NAMES]))
            if blendshapes:
                self._bs.append(blendshapes)
        if self.progress(t) < 1.0 or len(self._feats) < self.min_frames:
            return None
        return self._finish(t)

    def _finish(self, t: float) -> Baseline:
        arr = np.vstack(self._feats)
        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med), axis=0)
        bs = {}
        if self._bs:
            for k in BLENDSHAPE_KEYS:
                vals = [b[k] for b in self._bs if k in b]
                if vals:
                    bs[k] = float(np.median(vals))
        t0 = self._t0 if self._t0 is not None else t
        elapsed_min = max((t - t0) / 60.0, 1e-6)
        rate = self._blinks / elapsed_min if self._blinks >= 2 else None
        self.state = CalibrationState.COMPLETE
        return Baseline(
            features={k: float(v) for k, v in zip(FEATURE_NAMES, med)},
            feature_mad={k: float(v) for k, v in zip(FEATURE_NAMES, mad)},
            blendshapes=bs, blink_rate_per_min=rate, frames=len(self._feats),
        )
