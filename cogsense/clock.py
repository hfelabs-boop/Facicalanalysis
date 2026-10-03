"""Unified time base for the multimodal sync bus (FR-4.1).

All streams are stamped on one monotonic clock (``time.perf_counter``) at
*capture* time, so processing latency never turns into inter-stream drift.
UTC and LSL timestamps are derived through offsets that are re-estimated
periodically and slewed (never stepped) to keep drift well under 10 ms.
"""

from __future__ import annotations

import threading
import time

try:  # LSL is optional at runtime
    import pylsl
except Exception:  # pragma: no cover - environment dependent
    pylsl = None


class SyncClock:
    def __init__(self, resync_interval_s: float = 10.0, max_slew_ms: float = 0.5):
        self._lock = threading.Lock()
        self.resync_interval_s = resync_interval_s
        self.max_slew_s = max_slew_ms / 1000.0
        mono = time.perf_counter()
        self._utc_offset = time.time() - mono
        self._lsl_offset = (pylsl.local_clock() - time.perf_counter()) if pylsl else 0.0
        self._last_sync = mono
        self.max_observed_drift_ms = 0.0

    def now(self) -> float:
        """Unified clock, seconds (monotonic)."""
        t = time.perf_counter()
        if t - self._last_sync >= self.resync_interval_s:
            self._resync(t)
        return t

    def _resync(self, mono: float) -> None:
        with self._lock:
            target = time.time() - mono
            err = target - self._utc_offset
            self.max_observed_drift_ms = max(self.max_observed_drift_ms, abs(err) * 1000.0)
            # Slew towards wall-clock rather than stepping, so timestamps stay monotonic.
            self._utc_offset += max(-self.max_slew_s, min(self.max_slew_s, err))
            if pylsl:
                self._lsl_offset = pylsl.local_clock() - time.perf_counter()
            self._last_sync = mono

    def drift_ms(self) -> float:
        """Current disagreement between derived UTC and the system wall clock."""
        mono = time.perf_counter()
        return (time.time() - (mono + self._utc_offset)) * 1000.0

    def to_utc_ms(self, t: float) -> int:
        return int(round((t + self._utc_offset) * 1000.0))

    def from_utc_ms(self, utc_ms: float) -> float:
        return utc_ms / 1000.0 - self._utc_offset

    def to_lsl(self, t: float) -> float:
        return t + self._lsl_offset

    def from_lsl(self, lsl_t: float) -> float:
        return lsl_t - self._lsl_offset
