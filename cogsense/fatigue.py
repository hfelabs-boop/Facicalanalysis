"""Fatigue indicators from eyelid closure (separate from the Mental Effort Score).

Evidence (see dashboard/docs.html, "Other facial signals"):
  * PERCLOS, the share of time the eyes are more than 80 % closed, was the most accurate of the
    ocular measures for performance lapses in sleep-deprivation work (Dinges et al. 1998).
  * A 2023 review finds it is not reliably affected in moderate drowsiness, older adults or aviation
    tasks, that definitions vary (window, threshold, sampling rate), and that cameras are limited by
    frame rate, glasses and reflections.
  * Blink duration lengthens with time on task (Benedetto et al. 2011) and is among the best
    indicators of sleepiness (Schleicher et al. 2008).

So this reports the components with their coverage instead of one invented "fatigue score", and has
no alert threshold unless one is configured from validation data.
"""

from __future__ import annotations

from collections import deque

from cogsense.blink import BlinkSample
from cogsense.config import CogSenseConfig


class FatigueTracker:
    def __init__(self, cfg: CogSenseConfig):
        self.cfg = cfg
        self._segments: deque[tuple[float, float, bool]] = deque()  # (t, dt, closed)
        self._blinks: deque[tuple[float, float]] = deque()  # (t, duration_ms)
        self._prev_t: float | None = None
        self._t_first: float | None = None
        self._last_alert = -1e9

    def mark_gap(self) -> None:
        """Tracking was lost: the next frame must not credit the gap as open or closed time."""
        self._prev_t = None

    def update(self, t: float, ear_ratio: float, blink: BlinkSample) -> tuple[dict, list[dict]]:
        cfg = self.cfg
        if self._t_first is None:
            self._t_first = t
        if self._prev_t is not None:
            dt = min(max(t - self._prev_t, 0.0), cfg.fatigue_max_dt_s)
            self._segments.append((t, dt, ear_ratio <= cfg.perclos_closed_openness))
        self._prev_t = t
        if blink.completed_ms is not None:
            self._blinks.append((t, blink.completed_ms))
        while self._segments and t - self._segments[0][0] > cfg.perclos_window_s:
            self._segments.popleft()
        while self._blinks and t - self._blinks[0][0] > cfg.perclos_window_s:
            self._blinks.popleft()

        valid = sum(dt for _, dt, _ in self._segments)
        closed = sum(dt for _, dt, c in self._segments if c)
        elapsed = min(t - self._t_first, cfg.perclos_window_s)
        coverage = min(valid / elapsed, 1.0) if elapsed > 0 else 0.0
        perclos = closed / valid if valid >= cfg.perclos_min_coverage_s else None
        durs = [d for _, d in self._blinks]
        out = {
            "perclos": None if perclos is None else round(perclos, 3),
            "window_s": cfg.perclos_window_s,
            "coverage": round(coverage, 3),
            "blink_count": len(durs),
            "mean_blink_ms": round(sum(durs) / len(durs), 1) if durs else None,
            "long_closures": sum(1 for d in durs if d >= cfg.microsleep_ms),
        }
        events: list[dict] = []
        if (cfg.perclos_alert is not None and perclos is not None and perclos >= cfg.perclos_alert
                and t - self._last_alert >= cfg.perclos_window_s):
            self._last_alert = t
            events.append({"type": "FATIGUE_PERCLOS", "perclos": round(perclos, 3), "coverage": round(coverage, 3)})
        return out, events
