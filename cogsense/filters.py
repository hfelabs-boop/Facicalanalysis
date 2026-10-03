"""Small streaming signal-processing helpers."""

from __future__ import annotations

import math
from collections import deque


class EMA:
    """Time-aware exponential moving average (handles irregular frame intervals)."""

    def __init__(self, tau_s: float):
        self.tau = tau_s
        self.value: float | None = None
        self._t: float | None = None

    def update(self, x: float, t: float) -> float:
        if self.value is None or self._t is None or self.tau <= 0:
            self.value = x
        else:
            dt = max(t - self._t, 0.0)
            a = 1.0 - math.exp(-dt / self.tau)
            self.value += a * (x - self.value)
        self._t = t
        return self.value

    def reset(self) -> None:
        self.value = None
        self._t = None


class TimeWindow:
    """Sliding window of (t, value) pairs bounded by duration."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.items: deque[tuple[float, float]] = deque()

    def add(self, t: float, v: float) -> None:
        self.items.append((t, v))
        self.trim(t)

    def trim(self, now: float) -> None:
        while self.items and now - self.items[0][0] > self.seconds:
            self.items.popleft()

    def values(self) -> list[float]:
        return [v for _, v in self.items]

    def mean(self) -> float | None:
        if not self.items:
            return None
        return sum(v for _, v in self.items) / len(self.items)

    def std(self) -> float:
        n = len(self.items)
        if n < 2:
            return 0.0
        m = self.mean()
        return math.sqrt(sum((v - m) ** 2 for _, v in self.items) / (n - 1))

    def value_at_or_before(self, t: float) -> float | None:
        best = None
        for ti, v in self.items:
            if ti <= t:
                best = v
            else:
                break
        return best

    def max(self) -> float | None:
        return max((v for _, v in self.items), default=None)

    def span(self) -> float:
        return self.items[-1][0] - self.items[0][0] if len(self.items) > 1 else 0.0

    def clear(self) -> None:
        self.items.clear()


def clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x
