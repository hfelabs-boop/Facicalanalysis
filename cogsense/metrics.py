"""Cognitive state derivations (FR-3.1 – FR-3.3).

Mental Effort Score (MES, 0–100)
    Per frame:  e_t = w·AU4_t + (1 − w)·S_t,   S_t = clip(1 − BR_t / BR_baseline, 0, 1)
    where BR is the rolling blink rate (blinks/min) and S_t is blink-rate
    suppression relative to the operator's calibrated resting rate.
    MES_t = 100 · mean(e) over a rolling 5-second window.  (w = 0.7 by default.)
    If no blink rate is available yet, e_t = AU4_t.

Cognitive Friction Index (CFI, 0–1, continuous)
    CFI_t = clip(AU4_t · (0.6 + 0.4·C_t) + 0.15·R_t, 0, 1), halved for 2 s after
    a task-completion UI event, where
      C_t = max(clip(AU14_t / 0.30), clip(saccade_rate / 3 Hz))   co-occurrence term
      R_t = clip(ΔAU4 over the last 200 ms / 0.25)                 spike-rise term
    A discrete CFI *event* fires when an AU4 spike (> 0.65, rising ≥ 0.25 within
    200 ms) co-occurs with AU14 activation or rapid saccades and no task
    completion was logged in the preceding 2 s.

Automation surprise (FR-3.3)
    mean(AU1, AU2) > 0.50 followed within 800 ms by an AU4 surge
    (AU4 ≥ 0.50 and rising ≥ 0.20 within 200 ms). The flag latches for 1 s.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cogsense.au_engine import AUIntensities
from cogsense.config import CogSenseConfig
from cogsense.filters import TimeWindow, clip01


@dataclass
class FusionSignals:
    saccade_rate_hz: float = 0.0
    seconds_since_task_complete: float | None = None
    active_aoi: str | None = None


@dataclass
class MetricsSample:
    mental_effort_score: float
    cognitive_friction_index: float
    automation_surprise_flag: bool
    blink_suppression: float | None
    events: list[dict] = field(default_factory=list)


class CognitiveMetrics:
    def __init__(self, cfg: CogSenseConfig):
        self.cfg = cfg
        self._effort = TimeWindow(cfg.mes_window_s)
        self._au4 = TimeWindow(max(cfg.cfi_rise_window_s, 0.2) + 0.05)
        self._spike_until = -1e9
        self._last_cfi_event = -1e9
        self._brow_raise_t = -1e9
        self._surprise_until = -1e9
        self._cfi_peak = 0.0

    def _rise(self, t: float, au4: float, window: float) -> float:
        lo = min((v for ti, v in self._au4.items if t - ti <= window), default=au4)
        return au4 - lo

    def update(self, t: float, au: AUIntensities, blink_rate: float | None, baseline_blink_rate: float | None,
               fusion: FusionSignals) -> MetricsSample:
        cfg = self.cfg
        self._au4.add(t, au.au04)
        events: list[dict] = []

        # --- MES ---
        suppression = None
        if blink_rate is not None and baseline_blink_rate:
            suppression = clip01(1.0 - blink_rate / baseline_blink_rate)
            effort = cfg.mes_au4_weight * au.au04 + (1 - cfg.mes_au4_weight) * suppression
        else:
            effort = au.au04
        self._effort.add(t, effort)
        mes = 100.0 * (self._effort.mean() or 0.0)

        # --- CFI ---
        rise = self._rise(t, au.au04, cfg.cfi_rise_window_s)
        sacc = clip01(fusion.saccade_rate_hz / cfg.cfi_saccade_rate_hz) if cfg.cfi_saccade_rate_hz else 0.0
        co = max(clip01(au.au14 / cfg.cfi_au14_threshold), sacc)
        cfi = clip01(au.au04 * (0.6 + 0.4 * co) + 0.15 * clip01(rise / cfg.cfi_rise_min))
        task_recent = (fusion.seconds_since_task_complete is not None
                       and fusion.seconds_since_task_complete <= cfg.cfi_task_completion_window_s)
        if task_recent:
            cfi *= 0.5

        if au.au04 > cfg.cfi_au4_threshold and rise >= cfg.cfi_rise_min:
            self._spike_until = t + 0.3  # allow co-occurring cues to lag the spike slightly
            self._cfi_peak = 0.0
        if t <= self._spike_until:
            self._cfi_peak = max(self._cfi_peak, cfi)
            co_au14 = au.au14 >= cfg.cfi_au14_threshold
            co_sacc = fusion.saccade_rate_hz >= cfg.cfi_saccade_rate_hz
            if (co_au14 or co_sacc) and not task_recent and t - self._last_cfi_event >= cfg.cfi_event_refractory_s:
                self._last_cfi_event = t
                self._spike_until = -1e9
                events.append({
                    "type": "CFI_EVENT", "cfi": round(self._cfi_peak, 3), "au04": round(au.au04, 3),
                    "trigger": "AU14" if co_au14 else "SACCADES", "aoi": fusion.active_aoi,
                })

        # --- Automation surprise ---
        if (au.au01 + au.au02) / 2 > cfg.surprise_brow_threshold:
            self._brow_raise_t = t
        surge = au.au04 >= cfg.surprise_au4_surge and self._rise(t, au.au04, 0.2) >= 0.2
        if surge and 0 < t - self._brow_raise_t <= cfg.surprise_window_s and t > self._surprise_until:
            self._surprise_until = t + cfg.surprise_latch_s
            self._brow_raise_t = -1e9
            events.append({"type": "AUTOMATION_SURPRISE", "au04": round(au.au04, 3), "aoi": fusion.active_aoi})

        return MetricsSample(round(mes, 2), round(cfi, 3), t <= self._surprise_until, suppression, events)
