import numpy as np
import pytest

from cogsense.config import CogSenseConfig
from cogsense.engine import CogSenseEngine, FaceObservation
from cogsense.synthetic import FaceState, render


class Driver:
    """Feeds synthetic frames to an engine at a fixed rate."""

    def __init__(self, engine=None, fps=60.0, noise=0.0, seed=0):
        self.engine = engine or CogSenseEngine(CogSenseConfig())
        self.fps = fps
        self.t = 0.0
        self.frame = 0
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def step(self, state=None, face=True, **kw):
        st = state or FaceState(**kw)
        lm = render(st, noise_px=self.noise, rng=self.rng) if face else None
        p = self.engine.process(FaceObservation(self.t, lm, 1920, 1080, self.frame, int(1_700_000_000_000 + self.t * 1000)))
        self.t += 1.0 / self.fps
        self.frame += 1
        return p

    def hold(self, seconds, **kw):
        p = None
        for _ in range(int(round(seconds * self.fps))):
            p = self.step(**kw)
        return p

    def warmup(self, seconds=1.0):
        return self.hold(seconds)


@pytest.fixture
def driver():
    d = Driver()
    d.warmup()
    return d
