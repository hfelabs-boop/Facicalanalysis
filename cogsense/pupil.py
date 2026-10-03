"""Pupil-diameter tracking from the eye tracker (not the face camera).

Evidence (see dashboard/docs.html, "Other facial signals"): the task-evoked pupillary response tracks
cognitive load across memory, arithmetic, comprehension and vigilance tasks, but it is small (usually
under 0.5 mm) and returns to baseline within seconds (Beatty & Lucero-Wagoner 2000). A visible-light
RGB camera cannot resolve the pupil of a dark iris, so diameter has to come from an infrared tracker.

The pupil also responds to light. This module therefore reports change from a baseline taken under the
same display conditions, and if the sender also reports display luminance it flags the reading as
unreliable when luminance has shifted from the baseline.  It is a separate stream: it is NOT mixed into
the Mental Effort Score, because no study fixes how to weight it against the facial cues.
"""

from __future__ import annotations

import math
from statistics import median

from cogsense.config import CogSenseConfig
from cogsense.fusion import FusionHub


class PupilTracker:
    def __init__(self, cfg: CogSenseConfig, fusion: FusionHub):
        self.cfg = cfg
        self.fusion = fusion
        self.baseline_mm: float | None = None
        self.baseline_luma: float | None = None
        self.baseline_source: str | None = None  # "provisional" | "calibration"
        self._calibrating = False
        self._calib: list[tuple[float, float | None]] = []
        self._prov: list[tuple[float, float | None]] = []
        self._t_first: float | None = None

    # -- calibration hooks, driven by the engine's 15 s neutral baseline
    def start_calibration(self) -> None:
        self._calibrating = True
        self._calib = []

    def finish_calibration(self) -> None:
        self._calibrating = False
        if len(self._calib) >= 10:
            self._set_baseline(self._calib, "calibration")

    def _set_baseline(self, rows: list[tuple[float, float | None]], source: str) -> None:
        self.baseline_mm = median(d for d, _ in rows)
        lumas = [l for _, l in rows if l is not None]
        self.baseline_luma = median(lumas) if lumas else None
        self.baseline_source = source

    def update(self, t: float) -> dict | None:
        """Smoothed diameter and change from baseline at time ``t``; None when there is no fresh data."""
        cfg = self.cfg
        rows = [s for s in self.fusion.pupil_between(t - cfg.pupil_smooth_s, t)
                if s.valid and s.diameter_mm is not None and math.isfinite(s.diameter_mm) and s.diameter_mm > 0]
        if not rows:
            return None
        d = sum(s.diameter_mm for s in rows) / len(rows)
        lumas = [s.luminance for s in rows if s.luminance is not None]
        luma = sum(lumas) / len(lumas) if lumas else None

        if self._calibrating:
            self._calib.append((d, luma))
        if self.baseline_source != "calibration":
            if self._t_first is None:
                self._t_first = t
            if self.baseline_source is None or t - self._t_first <= cfg.pupil_provisional_s:
                self._prov.append((d, luma))
            if self.baseline_source is None and t - self._t_first >= cfg.pupil_provisional_s and len(self._prov) >= 10:
                self._set_baseline(self._prov, "provisional")

        shift = None
        if luma is not None and self.baseline_luma:
            shift = (luma - self.baseline_luma) / self.baseline_luma
        reliable = self.baseline_mm is not None and (shift is None or abs(shift) <= cfg.pupil_luma_tolerance)
        change = None if self.baseline_mm is None else d - self.baseline_mm
        return {
            "diameter_mm": round(d, 3),
            "baseline_mm": None if self.baseline_mm is None else round(self.baseline_mm, 3),
            "baseline_source": self.baseline_source,
            "change_mm": None if change is None else round(change, 3),
            "change_pct": None if change is None else round(100.0 * change / self.baseline_mm, 2),
            "luminance_shift": None if shift is None else round(shift, 3),
            "reliable": reliable,
        }
