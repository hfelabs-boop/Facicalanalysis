"""Blink (AU45) and sustained eye closure (AU43) detection with hysteresis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class BlinkSample:
    closed: bool
    onset: bool  # a new closure started this frame
    closure_ms: float  # duration of the current closure (0 if open)
    last_blink_ms: float | None
    rate_per_min: float | None
    microsleep: bool  # AU43: closure exceeded the microsleep threshold
    completed_ms: float | None = None  # duration of the closure that ended on this frame


class BlinkDetector:
    def __init__(self, close_ratio: float = 0.45, open_ratio: float = 0.60,
                 rate_window_s: float = 30.0, microsleep_ms: float = 500.0):
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.rate_window_s = rate_window_s
        self.microsleep_ms = microsleep_ms
        self.closed = False
        self._closed_since: float | None = None
        self._onsets: deque[float] = deque()
        self._t_first: float | None = None
        self.last_blink_ms: float | None = None

    def update(self, t: float, ear_ratio: float) -> BlinkSample:
        """``ear_ratio`` is the current eye-aspect ratio divided by its baseline."""
        if self._t_first is None:
            self._t_first = t
        onset = False
        completed: float | None = None
        if not self.closed and ear_ratio < self.close_ratio:
            self.closed, onset = True, True
            self._closed_since = t
            self._onsets.append(t)
        elif self.closed and ear_ratio > self.open_ratio:
            self.closed = False
            if self._closed_since is not None:
                self.last_blink_ms = completed = (t - self._closed_since) * 1000.0
            self._closed_since = None
        while self._onsets and t - self._onsets[0] > self.rate_window_s:
            self._onsets.popleft()
        closure_ms = (t - self._closed_since) * 1000.0 if self.closed and self._closed_since is not None else 0.0
        observed = min(t - self._t_first, self.rate_window_s)
        rate = len(self._onsets) * 60.0 / observed if observed >= 5.0 else None
        return BlinkSample(self.closed, onset, closure_ms, self.last_blink_ms, rate,
                           closure_ms >= self.microsleep_ms, completed)

    def reset_gap(self) -> None:
        """Called when tracking is lost so a closure doesn't span the gap."""
        self.closed = False
        self._closed_since = None
