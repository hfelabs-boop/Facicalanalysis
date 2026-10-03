"""AU regression core: baseline-relative geometric evidence, optionally fused
with learned blendshape evidence, mapped to 0–1 intensities (FR-1.1, FR-1.2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cogsense.calibration import Baseline
from cogsense.config import CogSenseConfig
from cogsense.features import FEATURE_NAMES
from cogsense.filters import EMA, clip01

AU_KEYS = ("au01", "au02", "au04", "au07", "au14", "au24")


@dataclass
class AUIntensities:
    au01: float
    au02: float
    au04: float
    au07: float
    au14: float
    au24: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in AU_KEYS}


def geometric_evidence(f: dict[str, float], base: Baseline, cfg: CogSenseConfig) -> dict[str, float]:
    """Map feature deltas from the neutral baseline to per-AU evidence in [0, 1]."""
    b = base.features
    s = cfg.scales
    # .get(): baselines saved before a feature existed still work (delta 0 = no evidence)
    d = {k: f[k] - b.get(k, f[k]) for k in FEATURE_NAMES}

    inner_drop = -(d["brow_inner_h_r"] + d["brow_inner_h_l"]) / 2
    gap_drop = -d["brow_gap"]
    outer_rise = (d["brow_outer_h_r"] + d["brow_outer_h_l"]) / 2

    ear_base = max((b["ear_r"] + b["ear_l"]) / 2, 1e-6)
    ear_drop_ratio = 1.0 - ((f["ear_r"] + f["ear_l"]) / 2) / ear_base
    lid_rise = -(d["lower_lid_r"] + d["lower_lid_l"]) / 2
    rest_lip = b.get("lip_thickness") or 0.0
    lip_thinning = (1.0 - f["lip_thickness"] / rest_lip) if rest_lip > 1e-6 else 0.0

    return {
        "au04": clip01(0.6 * inner_drop / s.brow_height + 0.4 * gap_drop / s.brow_gap),
        "au01": clip01(-inner_drop / s.brow_height),
        "au02": clip01(outer_rise / s.outer_brow_height),
        "au07": clip01(0.5 * ear_drop_ratio / s.eye_aperture_ratio + 0.5 * lid_rise / s.lower_lid_raise),
        "au14": clip01(0.6 * d["lip_corner_depth"] / s.lip_corner_depth + 0.4 * d["mouth_width"] / s.mouth_width),
        # Lips pressed together (AU24): visible lips thinned relative to this person's rest. The closed-mouth
        # and not-smiling conditions are applied after fusion by lip_press_gate().
        "au24": clip01(max(lip_thinning, 0.0) / s.lip_thinning_ratio),
    }


def lip_press_gate(f: dict[str, float], base: Baseline, cfg: CogSenseConfig) -> float:
    """0–1 factor that suppresses AU24 evidence unless the lips are actually together and not stretched.

    * An open mouth (talking, jaw drop, laughing) cannot be a lip press, whatever the blendshape says. On a
      real face the ``mouthPress`` blendshape already reads 0.10–0.16 with the mouth open and smiling.
    * A closed-lip smile also thins the lips, but it widens the mouth; a widening beyond the AU14 scale
      discounts the evidence to zero.
    """
    s, b = cfg.scales, base.features
    opening = max(f["mouth_aperture"] - b.get("mouth_aperture", f["mouth_aperture"]), 0.0)
    widening = max(f["mouth_width"] - b.get("mouth_width", f["mouth_width"]), 0.0)
    opening = max(opening - s.lip_open_deadzone, 0.0)  # below the dead-zone it is landmark jitter, not an opening
    return clip01(1.0 - opening / s.lip_open_gate) * clip01(1.0 - widening / s.mouth_width)


def _bs_rel(bs: dict[str, float], base: dict[str, float], *keys: str) -> float:
    vals = []
    for k in keys:
        if k in bs:
            rest = base.get(k, 0.0)
            vals.append(clip01((bs[k] - rest) / max(1.0 - rest, 1e-6)))
    return float(np.mean(vals)) if vals else float("nan")


def blendshape_evidence(bs: dict[str, float], base: Baseline) -> dict[str, float]:
    """MediaPipe ARKit-style blendshapes → AU evidence (baseline-subtracted)."""
    rb = base.blendshapes
    return {
        "au04": _bs_rel(bs, rb, "browDownLeft", "browDownRight"),
        "au01": _bs_rel(bs, rb, "browInnerUp"),
        "au02": _bs_rel(bs, rb, "browOuterUpLeft", "browOuterUpRight"),
        "au07": _bs_rel(bs, rb, "eyeSquintLeft", "eyeSquintRight"),
        "au14": _bs_rel(bs, rb, "mouthDimpleLeft", "mouthDimpleRight"),
        "au24": _bs_rel(bs, rb, "mouthPressLeft", "mouthPressRight"),
    }


class LinearAUModel:
    """Optional trained per-AU logistic model over baseline-relative features.

    Weights are produced by :mod:`cogsense.validation` from FACS-coded data
    (e.g. DISFA/BP4D). When loaded, its output replaces the hand-tuned
    geometric evidence for the AUs it covers.
    """

    def __init__(self, weights: dict[str, dict]):
        self.weights = weights

    @classmethod
    def load(cls, path: str | Path) -> "LinearAUModel":
        return cls(json.loads(Path(path).read_text()))

    def predict(self, f: dict[str, float], base: Baseline) -> dict[str, float]:
        x = np.array([f[k] - base.features[k] for k in FEATURE_NAMES])
        out = {}
        for au, w in self.weights.items():
            z = float(np.dot(np.asarray(w["coef"]), (x - np.asarray(w["mean"])) / np.asarray(w["scale"])) + w["intercept"])
            out[au] = 1.0 / (1.0 + np.exp(-z))
        return out


class AURegressor:
    def __init__(self, cfg: CogSenseConfig, model: LinearAUModel | None = None):
        self.cfg = cfg
        self.model = model
        self._ema = {k: EMA(cfg.lip_press_tau_s if k == "au24" else cfg.smoothing_tau_s) for k in AU_KEYS}
        self._held_au7 = 0.0

    def reset(self) -> None:
        for e in self._ema.values():
            e.reset()

    def update(self, t: float, feats: dict[str, float], base: Baseline, blendshapes: dict[str, float] | None,
               eyes_closed: bool, speech_active: bool) -> AUIntensities:
        ev = geometric_evidence(feats, base, self.cfg)
        if self.model is not None:
            ev.update(self.model.predict(feats, base))
        if blendshapes and self.cfg.blendshape_weight > 0:
            bev = blendshape_evidence(blendshapes, base)
            w = self.cfg.blendshape_weight
            for k, v in bev.items():
                if not np.isnan(v):
                    ev[k] = (1 - w) * ev[k] + w * v

        ev["au24"] *= lip_press_gate(feats, base, self.cfg)

        # A blink collapses the eye aperture; don't report it as lid tightening.
        if eyes_closed:
            ev["au07"] = self._held_au7
        else:
            self._held_au7 = ev["au07"]
        if speech_active:  # lower-face AUs are unreliable while the mouth is moving for speech or chewing
            ev["au14"] *= self.cfg.speech_lower_face_attenuation
            ev["au24"] *= self.cfg.speech_lower_face_attenuation

        out = {k: clip01(self._ema[k].update(ev[k], t)) for k in AU_KEYS}
        return AUIntensities(**out)
