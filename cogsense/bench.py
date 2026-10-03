"""Latency benchmark for the NFR budget (≤ 25 ms landmark + AU scoring per frame)."""

from __future__ import annotations

import statistics
import time

import numpy as np

from cogsense.engine import CogSenseEngine, FaceObservation
from cogsense.synthetic import FaceState, render


def _stats(ms: list[float]) -> dict:
    ms = sorted(ms)
    return {"mean_ms": round(statistics.fmean(ms), 3), "p50_ms": round(ms[len(ms) // 2], 3),
            "p95_ms": round(ms[int(len(ms) * 0.95) - 1], 3), "max_ms": round(ms[-1], 3)}


def run_bench(frames: int = 600, model: str | None = None, mediapipe: bool = False) -> dict:
    rng = np.random.default_rng(0)
    engine = CogSenseEngine()
    obs = [render(FaceState(au4=0.5 * (1 + np.sin(i / 20))), noise_px=0.3, rng=rng) for i in range(frames)]
    times = []
    for i, lm in enumerate(obs):
        t = time.perf_counter()
        engine.process(FaceObservation(i / 60, lm, 1920, 1080, i))
        times.append((time.perf_counter() - t) * 1000)
    out = {"engine": _stats(times[60:])}
    if mediapipe:
        from cogsense.runtime import ContrastNormalizer, MediaPipeTracker, fetch_model

        tracker = MediaPipeTracker(model or fetch_model())
        norm = ContrastNormalizer()
        img = (rng.random((1080, 1920, 3)) * 255).astype(np.uint8)
        mp_t, cl_t = [], []
        for i in range(min(frames, 120)):
            t = time.perf_counter()
            rgb, _ = norm(img)
            cl_t.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            tracker(rgb, i / 60)
            mp_t.append((time.perf_counter() - t) * 1000)
        tracker.close()
        out["contrast_normalization_1080p"] = _stats(cl_t)
        out["mediapipe_landmarker_1080p"] = _stats(mp_t)
    return out
