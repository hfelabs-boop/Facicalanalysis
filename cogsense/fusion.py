"""Multimodal fusion with eye tracking, RULA posture and C2 UI events (FR-4.2, FR-4.3).

External streams push timestamped samples here (via WebSocket, LSL or
direct API). For each face frame the engine asks for the nearest-in-time,
non-stale context and gets back the active AOI, posture scores, saccade
rate, compound-risk flags and a correlated-insight label.
"""

from __future__ import annotations

import bisect
import threading
from collections import deque
from dataclasses import dataclass

from cogsense.config import CogSenseConfig


@dataclass
class AOI:
    name: str
    x: float
    y: float
    w: float
    h: float

    def contains(self, gx: float, gy: float) -> bool:
        return self.x <= gx <= self.x + self.w and self.y <= gy <= self.y + self.h


@dataclass
class GazeSample:
    t: float
    x: float
    y: float
    saccade: bool = False


@dataclass
class RulaSample:
    t: float
    grand: int
    neck: int | None = None
    trunk: int | None = None


@dataclass
class PupilSample:
    t: float
    diameter_mm: float
    valid: bool = True
    luminance: float | None = None  # optional display luminance (any consistent unit)


@dataclass
class FusionContext:
    active_aoi: str | None
    gaze_x: float | None
    gaze_y: float | None
    saccade_rate_hz: float
    rula_grand_score: int | None
    rula_neck_score: int | None
    rula_trunk_score: int | None
    seconds_since_task_complete: float | None
    mission_phase: str | None


class _Timeline:
    """Time-sorted ring buffer supporting nearest-sample lookup."""

    def __init__(self, maxlen: int = 4096):
        self.ts: deque[float] = deque(maxlen=maxlen)
        self.items: deque = deque(maxlen=maxlen)

    def add(self, t: float, item) -> None:
        if self.ts and t < self.ts[-1]:  # rare out-of-order arrival: insert in place
            ts, items = list(self.ts), list(self.items)
            i = bisect.bisect_right(ts, t)
            ts.insert(i, t)
            items.insert(i, item)
            self.ts = deque(ts, maxlen=self.ts.maxlen)
            self.items = deque(items, maxlen=self.items.maxlen)
        else:
            self.ts.append(t)
            self.items.append(item)

    def nearest(self, t: float, max_age: float):
        if not self.ts:
            return None
        ts = self.ts
        i = bisect.bisect_left(ts, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts) and abs(ts[j] - t) <= max_age:
                if best is None or abs(ts[j] - t) < abs(ts[best] - t):
                    best = j
        return self.items[best] if best is not None else None

    def count_between(self, t0: float, t1: float, pred) -> int:
        return sum(1 for ti, it in zip(self.ts, self.items) if t0 < ti <= t1 and pred(it))


class FusionHub:
    """Thread-safe store of external context. Timestamps are on the unified clock (seconds)."""

    def __init__(self, cfg: CogSenseConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._gaze = _Timeline()
        self._rula = _Timeline(1024)
        self._task_completions: deque[float] = deque(maxlen=256)
        self._pupil = _Timeline(8192)
        self.aois: list[AOI] = []
        self.mission_phase: str | None = None

    def set_aois(self, aois: list[AOI]) -> None:
        with self._lock:
            self.aois = list(aois)

    def set_phase(self, name: str | None) -> None:
        with self._lock:
            self.mission_phase = name

    def add_gaze(self, s: GazeSample) -> None:
        with self._lock:
            self._gaze.add(s.t, s)

    def add_rula(self, s: RulaSample) -> None:
        with self._lock:
            self._rula.add(s.t, s)

    def add_pupil(self, s: PupilSample) -> None:
        with self._lock:
            self._pupil.add(s.t, s)

    def pupil_between(self, t0: float, t1: float) -> list[PupilSample]:
        with self._lock:
            return [it for ti, it in zip(self._pupil.ts, self._pupil.items) if t0 < ti <= t1]

    def add_task_completion(self, t: float) -> None:
        with self._lock:
            self._task_completions.append(t)

    def hit_test(self, gx: float, gy: float) -> str | None:
        # Later AOIs are drawn on top, so they win on overlap.
        for aoi in reversed(self.aois):
            if aoi.contains(gx, gy):
                return aoi.name
        return None

    def context(self, t: float) -> FusionContext:
        cfg = self.cfg
        with self._lock:
            gaze = self._gaze.nearest(t, cfg.gaze_staleness_s)
            rula = self._rula.nearest(t, cfg.rula_staleness_s)
            sacc = self._gaze.count_between(t - 1.0, t, lambda g: g.saccade)
            done = [tc for tc in self._task_completions if tc <= t]
            since = t - done[-1] if done else None
            aoi = self.hit_test(gaze.x, gaze.y) if gaze else None
            phase = self.mission_phase
        return FusionContext(
            active_aoi=aoi,
            gaze_x=gaze.x if gaze else None,
            gaze_y=gaze.y if gaze else None,
            saccade_rate_hz=float(sacc),
            rula_grand_score=rula.grand if rula else None,
            rula_neck_score=rula.neck if rula else None,
            rula_trunk_score=rula.trunk if rula else None,
            seconds_since_task_complete=since,
            mission_phase=phase,
        )


class CompoundRiskTracker:
    """Edge-triggers COMPOUND_POSTURE_RISK so a sustained condition logs one event, not one per frame."""

    def __init__(self, rearm_s: float = 1.0):
        self.rearm_s = rearm_s
        self.active = False
        self._clear_since: float | None = None
        self.onset_t: float | None = None

    def update(self, t: float, condition: bool) -> bool:
        """Returns True on the frame the condition (re)starts."""
        if condition:
            self._clear_since = None
            if not self.active:
                self.active, self.onset_t = True, t
                return True
            return False
        if self.active:
            if self._clear_since is None:
                self._clear_since = t
            elif t - self._clear_since >= self.rearm_s:
                self.active = False
        return False


def correlate(ctx: FusionContext, au04: float, au07: float, cfi: float, mes: float, surprise: bool,
              microsleep: bool, cfg: CogSenseConfig, t: float = 0.0,
              tracker: CompoundRiskTracker | None = None) -> tuple[str | None, list[dict]]:
    """Derive the correlated-insight label and compound-risk events for one frame."""
    events: list[dict] = []
    neck_high = ctx.rula_neck_score is not None and ctx.rula_neck_score >= cfg.rula_neck_threshold
    trunk_high = ctx.rula_trunk_score is not None and ctx.rula_trunk_score >= cfg.rula_trunk_threshold
    posture_high = neck_high or trunk_high
    visual_strain = au07 >= cfg.compound_au7_threshold
    cognitive_strain = au04 >= cfg.compound_au4_threshold

    compound = posture_high and (visual_strain or cognitive_strain)
    if tracker.update(t, compound) if tracker is not None else compound:
        events.append({
            "type": "COMPOUND_POSTURE_RISK",
            "rula_neck": ctx.rula_neck_score, "rula_trunk": ctx.rula_trunk_score,
            "au07": round(au07, 3), "au04": round(au04, 3), "aoi": ctx.active_aoi,
        })

    if microsleep:
        insight = "FATIGUE_MICROSLEEP_RISK"
    elif surprise:
        insight = "AUTOMATION_SURPRISE"
    elif posture_high and visual_strain:
        insight = "POSTURE_DRIVEN_VISUAL_COMPENSATION"
    elif cfi >= cfg.cfi_marker_threshold and ctx.active_aoi:
        insight = "HIGH_COGNITIVE_STRAIN_ON_COMPLEX_WIDGET"
    elif cfi >= cfg.cfi_marker_threshold:
        insight = "HIGH_COGNITIVE_FRICTION"
    elif visual_strain and not posture_high:
        insight = "VISUAL_STRAIN_DISPLAY_LEGIBILITY"
    elif posture_high and cognitive_strain:
        insight = "POSTURAL_LOAD_WITH_COGNITIVE_STRAIN"
    elif mes >= 60:
        insight = "SUSTAINED_MENTAL_EFFORT"
    else:
        insight = None
    return insight, events
