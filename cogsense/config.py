"""Tunable parameters. Defaults implement the thresholds in PRD v1.0.0."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class AUScales:
    """Geometric displacement (in inter-ocular-distance units) that maps to intensity 1.0."""

    brow_height: float = 0.08
    brow_gap: float = 0.10
    outer_brow_height: float = 0.09
    eye_aperture_ratio: float = 0.42  # relative EAR drop for full AU7
    lower_lid_raise: float = 0.05
    lip_corner_depth: float = 0.06
    mouth_width: float = 0.06


@dataclass
class CogSenseConfig:
    # --- sampling / calibration (FR-1.2, FR-1.3) ---
    target_hz: float = 60.0
    calibration_seconds: float = 15.0
    calibration_min_frames: int = 90

    # --- AU regression core ---
    scales: AUScales = field(default_factory=AUScales)
    blendshape_weight: float = 0.5  # weight of learned blendshape evidence when available
    smoothing_tau_s: float = 0.04  # EMA time constant; short enough to keep <500 ms transients

    # --- blink (AU43/45) ---
    blink_close_ratio: float = 0.45  # EAR / baseline EAR below which eyes are closed
    blink_open_ratio: float = 0.60
    blink_rate_window_s: float = 30.0
    microsleep_ms: float = 500.0
    default_blink_rate_per_min: float = 17.0

    # --- speech / mastication mask (FR-2.2) ---
    speech_window_s: float = 0.6
    speech_aperture_std: float = 0.018
    speech_open_delta: float = 0.07
    speech_lower_face_attenuation: float = 0.3
    speech_confidence_factor: float = 0.85

    # --- head pose invariance (FR-2.1) ---
    yaw_limit_deg: float = 30.0
    pitch_limit_deg: float = 20.0
    roll_limit_deg: float = 20.0
    pose_falloff_deg: float = 20.0  # confidence decays to 0 this far beyond the limit

    # --- tracking confidence / graceful fallback ---
    lost_confidence: float = 0.3
    locked_confidence: float = 0.6
    min_iod_px: float = 25.0

    # --- fatigue: PERCLOS-style eyelid closure (kept out of MES; see docs) ---
    perclos_window_s: float = 60.0  # window length is a convention, not a fixed standard
    perclos_closed_openness: float = 0.20  # eye counts as closed at <= 20 % of resting openness (">80 % closed")
    perclos_min_coverage_s: float = 10.0  # valid tracking needed before a value is reported
    perclos_alert: float | None = None  # no evidence-based cut-off: off unless set from validation data
    fatigue_max_dt_s: float = 0.1  # cap frame gaps so dropouts cannot inflate closure time

    # --- pupil (ingested from the eye tracker; a plain RGB camera cannot measure it) ---
    pupil_smooth_s: float = 0.5
    pupil_provisional_s: float = 5.0  # provisional baseline from the first seconds of valid data
    pupil_luma_tolerance: float = 0.10  # design choice: flag pupil unreliable if display luminance shifts > 10 %

    # --- cognitive metrics (FR-3.x) ---
    mes_window_s: float = 5.0
    mes_au4_weight: float = 0.7
    cfi_au4_threshold: float = 0.65
    cfi_rise_window_s: float = 0.2
    cfi_rise_min: float = 0.25
    cfi_au14_threshold: float = 0.30
    cfi_saccade_rate_hz: float = 3.0
    cfi_task_completion_window_s: float = 2.0
    cfi_event_refractory_s: float = 1.0
    cfi_marker_threshold: float = 0.70
    surprise_brow_threshold: float = 0.50
    surprise_window_s: float = 0.8
    surprise_au4_surge: float = 0.50
    surprise_latch_s: float = 1.0

    # --- fusion (FR-4.x) ---
    gaze_staleness_s: float = 0.25
    rula_staleness_s: float = 2.0
    rula_neck_threshold: int = 3
    rula_trunk_threshold: int = 3
    compound_au7_threshold: float = 0.40
    compound_au4_threshold: float = 0.50

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CogSenseConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known and k != "scales"}
        cfg = cls(**kwargs)
        if "scales" in data:
            cfg.scales = AUScales(**data["scales"])
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "CogSenseConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
