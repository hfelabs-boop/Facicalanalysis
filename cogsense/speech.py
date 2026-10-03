"""Speech & mastication rejection mask (FR-2.2).

Speech and chewing produce sustained lower-face oscillation (lip aperture
and jaw displacement) that biases AU14 and, through skin coupling, the
upper-face geometry. When detected, the lower-face AU is attenuated and
the frame confidence is reduced.
"""

from __future__ import annotations

from dataclasses import dataclass

from cogsense.filters import TimeWindow


@dataclass
class SpeechState:
    active: bool
    aperture_std: float
    open_delta: float


class SpeechMask:
    def __init__(self, window_s: float = 0.6, aperture_std: float = 0.018, open_delta: float = 0.07,
                 hold_s: float = 0.4):
        self.aperture = TimeWindow(window_s)
        self.jaw = TimeWindow(window_s)
        self.std_thresh = aperture_std
        self.open_thresh = open_delta
        self.hold_s = hold_s
        self._last_active: float | None = None

    def update(self, t: float, aperture: float, jaw: float, base_aperture: float, base_jaw: float) -> SpeechState:
        self.aperture.add(t, aperture)
        self.jaw.add(t, jaw)
        std = max(self.aperture.std(), self.jaw.std())
        open_delta = max(aperture - base_aperture, jaw - base_jaw)
        moving = std > self.std_thresh
        # A sustained open mouth with motion (talking) or rhythmic jaw motion (chewing).
        active = moving or (open_delta > self.open_thresh and std > self.std_thresh * 0.5)
        if active:
            self._last_active = t
        elif self._last_active is not None and t - self._last_active < self.hold_s:
            active = True
        return SpeechState(active, std, open_delta)
