"""CogSense processing core: one face observation in, one telemetry payload out.

The engine is pure computation (no I/O, no threads), which keeps it
deterministic and unit-testable. Camera capture, the landmark tracker and
the sync bus live in :mod:`cogsense.runtime`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from cogsense import landmarks as L
from cogsense.au_engine import AURegressor, LinearAUModel
from cogsense.blink import BlinkDetector
from cogsense.calibration import BLENDSHAPE_KEYS, Baseline, BaselineCalibrator, CalibrationState
from cogsense.config import CogSenseConfig
from cogsense.features import FEATURE_NAMES, extract
from cogsense.fatigue import FatigueTracker
from cogsense.filters import clip01
from cogsense.fusion import CompoundRiskTracker, FusionHub, correlate
from cogsense.geometry import normalize, to_pixel_space
from cogsense.metrics import CognitiveMetrics, FusionSignals
from cogsense.pupil import PupilTracker
from cogsense.speech import SpeechMask

PROVISIONAL_FRAMES = 30


@dataclass
class FaceObservation:
    timestamp_s: float  # unified-clock capture time (seconds)
    landmarks: np.ndarray | None  # (N, 3) MediaPipe-normalized, or None if no face
    width: int
    height: int
    frame_id: int = 0
    timestamp_utc_ms: int | None = None
    blendshapes: dict[str, float] | None = None
    luma_mean: float | None = None  # mean frame luminance 0–255 after normalization


def _pose_factor(value: float, limit: float, falloff: float) -> float:
    excess = abs(value) - limit
    return 1.0 if excess <= 0 else clip01(1.0 - excess / falloff)


class CogSenseEngine:
    def __init__(self, cfg: CogSenseConfig | None = None, fusion: FusionHub | None = None,
                 baseline: Baseline | None = None, au_model: LinearAUModel | None = None):
        self.cfg = cfg or CogSenseConfig()
        self.fusion = fusion or FusionHub(self.cfg)
        self.baseline = baseline
        self.au = AURegressor(self.cfg, au_model)
        self.blink = BlinkDetector(self.cfg.blink_close_ratio, self.cfg.blink_open_ratio,
                                   self.cfg.blink_rate_window_s, self.cfg.microsleep_ms)
        self.speech = SpeechMask(self.cfg.speech_window_s, self.cfg.speech_aperture_std, self.cfg.speech_open_delta)
        self.metrics = CognitiveMetrics(self.cfg)
        self.fatigue = FatigueTracker(self.cfg)
        self.pupil = PupilTracker(self.cfg, self.fusion)
        self.calibrator = BaselineCalibrator(self.cfg.calibration_seconds, self.cfg.calibration_min_frames)
        self._provisional: list[np.ndarray] = []
        self._provisional_bs: list[dict[str, float]] = []
        self._prev_points: np.ndarray | None = None
        self._compound = CompoundRiskTracker()
        self.last_payload: dict | None = None

    # ------------------------------------------------------------------ calibration
    def start_calibration(self, t: float) -> None:
        self.calibrator.start(t)
        self.pupil.start_calibration()

    @property
    def calibration_status(self) -> str:
        if self.calibrator.state is CalibrationState.COLLECTING:
            return "CALIBRATING"
        if self.baseline is None:
            return "UNCALIBRATED"
        return "PROVISIONAL" if self.baseline.is_default else "CALIBRATED"

    def _baseline_blink_rate(self) -> float:
        if self.baseline and self.baseline.blink_rate_per_min:
            return self.baseline.blink_rate_per_min
        return self.cfg.default_blink_rate_per_min

    # ------------------------------------------------------------------ processing
    def process(self, obs: FaceObservation) -> dict:
        t0 = time.perf_counter()
        utc_ms = obs.timestamp_utc_ms if obs.timestamp_utc_ms is not None else int(time.time() * 1000)
        lm = obs.landmarks
        if lm is None or len(lm) < L.MIN_LANDMARKS:
            return self._finish(self._lost_payload(obs, utc_ms, "NO_FACE"), t0)

        pts_px = to_pixel_space(lm, obs.width, obs.height)
        face = normalize(pts_px)
        feats = extract(face.points)
        cfg = self.cfg

        # ---------- tracking confidence ----------
        pose_ok = (abs(face.yaw_deg) <= cfg.yaw_limit_deg and abs(face.pitch_deg) <= cfg.pitch_limit_deg
                   and abs(face.roll_deg) <= cfg.roll_limit_deg)
        conf = (_pose_factor(face.yaw_deg, cfg.yaw_limit_deg, cfg.pose_falloff_deg)
                * _pose_factor(face.pitch_deg, cfg.pitch_limit_deg, cfg.pose_falloff_deg)
                * _pose_factor(face.roll_deg, cfg.roll_limit_deg, cfg.pose_falloff_deg))
        conf *= clip01(face.iod_px / (2 * cfg.min_iod_px)) if face.iod_px < 2 * cfg.min_iod_px else 1.0
        if self._prev_points is not None:
            jitter = float(np.mean(np.linalg.norm(face.points[:L.MIN_LANDMARKS] - self._prev_points, axis=1)))
            conf *= clip01(1.0 - (jitter - 0.08) / 0.25)
        self._prev_points = face.points[:L.MIN_LANDMARKS].copy()
        if feats["brow_inner_h_r"] <= 0 or feats["brow_inner_h_l"] <= 0:
            conf *= 0.3  # implausible geometry: brows below eyes → occlusion / tracker failure
        if obs.luma_mean is not None and not 25 <= obs.luma_mean <= 235:
            conf *= 0.6

        # ---------- provisional baseline (until a 15 s calibration is run) ----------
        if self.baseline is None:
            if conf >= cfg.locked_confidence:
                self._provisional.append(np.array([feats[k] for k in FEATURE_NAMES]))
                if obs.blendshapes:
                    self._provisional_bs.append(obs.blendshapes)
            if len(self._provisional) >= PROVISIONAL_FRAMES:
                med = np.median(np.vstack(self._provisional), axis=0)
                # Resting blendshape values matter too: without them a face that rests with lowered
                # brows would read as AU4 until the 15 s calibration is run.
                bs = {k: float(np.median([b[k] for b in self._provisional_bs if k in b]))
                      for k in BLENDSHAPE_KEYS if any(k in b for b in self._provisional_bs)}
                self.baseline = Baseline({k: float(v) for k, v in zip(FEATURE_NAMES, med)},
                                         blendshapes=bs, is_default=True, frames=len(self._provisional))
            else:
                payload = self._lost_payload(obs, utc_ms, "ACQUIRING", face=face, conf=conf)
                return self._finish(payload, t0)

        base = self.baseline
        ear_base = max((base.features["ear_r"] + base.features["ear_l"]) / 2, 1e-6)
        ear_ratio = ((feats["ear_r"] + feats["ear_l"]) / 2) / ear_base
        blink = self.blink.update(obs.timestamp_s, ear_ratio)
        speech = self.speech.update(obs.timestamp_s, feats["mouth_aperture"], feats["jaw_drop"],
                                    base.features["mouth_aperture"], base.features["jaw_drop"])
        if speech.active:
            conf *= cfg.speech_confidence_factor
        if not blink.closed:
            asym = abs(feats["ear_r"] - feats["ear_l"]) / max(feats["ear_r"], feats["ear_l"], 1e-6)
            if asym > 0.6:
                conf *= 0.5  # one eye hidden (hand, headset boom, glare)

        if self.calibrator.state is CalibrationState.COLLECTING:
            new_base = self.calibrator.add(obs.timestamp_s, feats, obs.blendshapes, blink.closed, blink.onset,
                                           usable=conf >= cfg.locked_confidence and not speech.active)
            if new_base is not None:
                self.baseline = new_base
                self.au.reset()
                self.pupil.finish_calibration()

        if conf < cfg.lost_confidence:
            self.au.reset()
            return self._finish(self._lost_payload(obs, utc_ms, "LOW_CONFIDENCE", face=face, conf=conf), t0)

        au = self.au.update(obs.timestamp_s, feats, self.baseline, obs.blendshapes, blink.closed, speech.active)
        fat, fat_events = self.fatigue.update(obs.timestamp_s, ear_ratio, blink)
        pupil = self.pupil.update(obs.timestamp_s)
        ctx = self.fusion.context(obs.timestamp_s)
        m = self.metrics.update(obs.timestamp_s, au, blink.rate_per_min, self._baseline_blink_rate(),
                                FusionSignals(ctx.saccade_rate_hz, ctx.seconds_since_task_complete, ctx.active_aoi))
        insight, fusion_events = correlate(ctx, au.au04, au.au07, m.cognitive_friction_index,
                                           m.mental_effort_score, m.automation_surprise_flag, blink.microsleep, cfg,
                                           obs.timestamp_s, self._compound)
        events = m.events + fusion_events + fat_events
        if blink.microsleep and blink.closure_ms - 1000.0 / cfg.target_hz < cfg.microsleep_ms:
            events.append({"type": "MICROSLEEP", "closure_ms": round(blink.closure_ms, 1)})

        payload = {
            "timestamp_utc_ms": utc_ms,
            "frame_id": obs.frame_id,
            "tracking_status": "LOCKED" if conf >= cfg.locked_confidence else "DEGRADED",
            "confidence": round(conf, 3),
            "head_pose": self._pose(face, pose_ok),
            "action_units": {
                "au04_brow_lowerer": round(au.au04, 3),
                "au07_lid_tightener": round(au.au07, 3),
                "au01_inner_brow_raiser": round(au.au01, 3),
                "au02_outer_brow_raiser": round(au.au02, 3),
                "au14_dimpler": round(au.au14, 3),
                "au45_blink_state": int(blink.closed),
                "au43_eyes_closed": bool(blink.microsleep),
            },
            "blink": {
                "rate_per_min": None if blink.rate_per_min is None else round(blink.rate_per_min, 1),
                "last_duration_ms": None if blink.last_blink_ms is None else round(blink.last_blink_ms, 1),
                "closure_ms": round(blink.closure_ms, 1),
            },
            "cognitive_metrics": {
                "mental_effort_score": m.mental_effort_score,
                "cognitive_friction_index": m.cognitive_friction_index,
                "automation_surprise_flag": m.automation_surprise_flag,
                "speech_interference_detected": speech.active,
            },
            "fatigue": fat,
            "fusion_context": self._fusion(ctx, insight, pupil),
            "calibration": {"status": self.calibration_status,
                            "progress": round(self.calibrator.progress(obs.timestamp_s), 3)},
            "events": events,
        }
        return self._finish(payload, t0)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pose(face, pose_ok: bool) -> dict:
        return {"yaw_deg": round(face.yaw_deg, 2), "pitch_deg": round(face.pitch_deg, 2),
                "roll_deg": round(face.roll_deg, 2), "within_operating_range": pose_ok}

    @staticmethod
    def _fusion(ctx, insight: str | None, pupil: dict | None = None) -> dict:
        return {
            "active_aoi": ctx.active_aoi,
            "gaze_px": None if ctx.gaze_x is None else [round(ctx.gaze_x, 1), round(ctx.gaze_y, 1)],
            "saccade_rate_hz": ctx.saccade_rate_hz,
            "rula_grand_score": ctx.rula_grand_score,
            "rula_neck_score": ctx.rula_neck_score,
            "rula_trunk_score": ctx.rula_trunk_score,
            "mission_phase": ctx.mission_phase,
            "pupil": pupil,
            "correlated_insight": insight,
        }

    def _lost_payload(self, obs: FaceObservation, utc_ms: int, reason: str, face=None, conf: float = 0.0) -> dict:
        """Graceful fallback: explicit low-confidence flag, never stale or hallucinated AUs."""
        self.fatigue.mark_gap()  # a gap must never count as open or closed time
        if face is None:
            self.blink.reset_gap()
            self.au.reset()
            self._prev_points = None
        ctx = self.fusion.context(obs.timestamp_s)
        return {
            "timestamp_utc_ms": utc_ms,
            "frame_id": obs.frame_id,
            "tracking_status": "ACQUIRING" if reason == "ACQUIRING" else "LOST",
            "tracking_loss_reason": reason,
            "confidence": round(min(conf, self.cfg.lost_confidence - 1e-3) if reason != "ACQUIRING" else conf, 3),
            "head_pose": self._pose(face, False) if face is not None else None,
            "action_units": None,
            "blink": None,
            "cognitive_metrics": None,
            "fatigue": None,
            "fusion_context": self._fusion(ctx, None),
            "calibration": {"status": self.calibration_status,
                            "progress": round(self.calibrator.progress(obs.timestamp_s), 3)},
            "events": [],
        }

    def _finish(self, payload: dict, t0: float) -> dict:
        payload["processing_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        self.last_payload = payload
        return payload
