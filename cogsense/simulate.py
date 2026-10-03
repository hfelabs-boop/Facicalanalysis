"""Scripted C2-console scenario that drives the real engine with synthetic faces,
gaze and RULA streams. Used for demos, dashboard development and
end-to-end tests without a camera or operator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from cogsense.config import CogSenseConfig
from cogsense.engine import CogSenseEngine, FaceObservation
from cogsense.fusion import AOI, FusionHub, GazeSample, PupilSample, RulaSample
from cogsense.synthetic import FaceState, render

W, H = 1920, 1080

CONSOLE_AOIS = [
    AOI("TACTICAL_RADAR_WIDGET_PRIMARY", 40, 60, 1100, 700),
    AOI("TRACK_TABLE", 1180, 60, 700, 420),
    AOI("ASSET_STATUS_PANEL", 1180, 500, 700, 260),
    AOI("ENGAGEMENT_MENU_NESTED", 40, 800, 700, 240),
    AOI("COMMS_LOG", 780, 800, 1100, 240),
]


@dataclass
class Phase:
    name: str
    duration_s: float
    load: float  # baseline AU4 drive 0..1
    aoi_weights: dict[str, float]
    rula: tuple[int, int, int]  # grand, neck, trunk
    friction_rate: float = 0.0  # friction spikes per minute
    surprise_rate: float = 0.0
    squint: float = 0.0
    talk_fraction: float = 0.0
    occlusion_s: float = 0.0
    drowsy: float = 0.0  # 0..1: longer, more frequent eyelid closures (late-shift fatigue)
    pupil_gain_mm: float = 0.8  # task-evoked dilation per unit of load (real responses are usually < 0.5 mm)


DEFAULT_SCENARIO = [
    Phase("CALIBRATION", 16, 0.0, {"COMMS_LOG": 1}, (2, 1, 1)),
    Phase("TRANSIT", 40, 0.15, {"TACTICAL_RADAR_WIDGET_PRIMARY": 2, "TRACK_TABLE": 1, "COMMS_LOG": 1}, (3, 2, 2),
          talk_fraction=0.15),
    Phase("SWARM_ENGAGEMENT", 60, 0.55, {"TACTICAL_RADAR_WIDGET_PRIMARY": 5, "ENGAGEMENT_MENU_NESTED": 2, "TRACK_TABLE": 2},
          (4, 2, 2), friction_rate=6, occlusion_s=2.0),
    Phase("AUTOMATION_HANDOFF", 40, 0.35, {"ASSET_STATUS_PANEL": 3, "TACTICAL_RADAR_WIDGET_PRIMARY": 2}, (4, 2, 3),
          surprise_rate=4),
    Phase("DEGRADED_DISPLAY", 40, 0.3, {"TRACK_TABLE": 4, "COMMS_LOG": 1}, (5, 3, 3), squint=0.6, friction_rate=2),
    Phase("LATE_SHIFT", 60, 0.2, {"TRACK_TABLE": 3, "COMMS_LOG": 2}, (4, 3, 3), drowsy=1.0),
]


class ScenarioSimulator:
    def __init__(self, phases: list[Phase] | None = None, fps: float = 30.0, seed: int = 1,
                 cfg: CogSenseConfig | None = None, start_utc_ms: int | None = None):
        self.phases = phases or DEFAULT_SCENARIO
        self.fps = fps
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg or CogSenseConfig()
        self.fusion = FusionHub(self.cfg)
        self.fusion.set_aois(CONSOLE_AOIS)
        self.engine = CogSenseEngine(self.cfg, self.fusion)
        self.start_utc_ms = start_utc_ms if start_utc_ms is not None else int(time.time() * 1000)
        self.duration_s = sum(p.duration_s for p in self.phases)

    def _aoi_center(self, name: str) -> tuple[float, float]:
        a = next(a for a in CONSOLE_AOIS if a.name == name)
        return a.x + a.w / 2, a.y + a.h / 2

    def run(self) -> Iterator[dict]:
        rng, dt = self.rng, 1.0 / self.fps
        t, frame = 0.0, 0
        gaze_target = self._aoi_center("COMMS_LOG")
        gaze = np.array(gaze_target)
        next_saccade = 0.0
        next_blink = 2.0
        blink_until = -1.0
        events: list[tuple[float, str]] = []  # (start, kind)
        au4_slow = 0.0
        yaw = pitch = 0.0

        for pi, ph in enumerate(self.phases):
            self.fusion.set_phase(ph.name)
            if ph.name == "CALIBRATION":
                self.engine.start_calibration(t)
            phase_end = t + ph.duration_s
            occl_start = t + ph.duration_s * 0.5 if ph.occlusion_s else None
            names, weights = zip(*ph.aoi_weights.items())
            probs = np.array(weights, float) / sum(weights)
            talking_until = -1.0
            while t < phase_end:
                # --- scripted transients ---
                if ph.friction_rate and rng.random() < ph.friction_rate / 60 * dt:
                    events.append((t, "friction"))
                if ph.surprise_rate and rng.random() < ph.surprise_rate / 60 * dt:
                    events.append((t, "surprise"))
                if ph.talk_fraction and t > talking_until and rng.random() < ph.talk_fraction / 3 * dt:
                    talking_until = t + rng.uniform(1.5, 4.0)
                events = [e for e in events if t - e[0] < 2.0]

                au4_slow += (ph.load - au4_slow) * (1 - math.exp(-dt / 4.0))
                st = FaceState(au4=max(0.0, au4_slow + 0.05 * math.sin(t * 0.7)), au7=ph.squint * (0.8 + 0.2 * math.sin(t)))
                saccade_burst = False
                for start, kind in events:
                    age = t - start
                    if kind == "friction":
                        # 150 ms onset to a ~0.85 AU4 peak with AU14 annoyance, then decay
                        env = min(age / 0.15, 1.0) * math.exp(-max(age - 0.6, 0) / 0.4)
                        st.au4 = max(st.au4, 0.88 * env)
                        st.au14 = max(st.au14, 0.55 * env)
                        st.au24 = max(st.au24, 0.6 * env)
                        saccade_burst = saccade_burst or age < 1.0
                    else:
                        if age < 0.4:
                            st.au1 = st.au2 = 0.8 * min(age / 0.1, 1.0)
                        else:
                            env = min((age - 0.4) / 0.12, 1.0) * math.exp(-max(age - 1.0, 0) / 0.4)
                            st.au4 = max(st.au4, 0.8 * env)
                st.au4 = min(st.au4, 1.0)

                # blinks: suppressed under load
                if t >= next_blink:
                    blink_until = t + rng.uniform(0.12, 0.25) + ph.drowsy * rng.uniform(0.2, 0.5)
                    rate = 18.0 * (1.0 - 0.6 * ph.load) + ph.drowsy * 10.0
                    next_blink = t + rng.exponential(60.0 / rate) + 0.3
                if t < blink_until:
                    st.blink = 1.0
                if t < talking_until:
                    st.mouth_open = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 4.0)

                yaw += (rng.normal(0, 3) - yaw) * 0.02
                pitch += ((-8 if ph.rula[1] >= 3 else 0) + rng.normal(0, 2) - pitch) * 0.02
                st.yaw, st.pitch = yaw, pitch

                # --- gaze stream ---
                is_sacc = False
                if t >= next_saccade:
                    target = names[rng.choice(len(names), p=probs)]
                    cx, cy = self._aoi_center(target)
                    a = next(a for a in CONSOLE_AOIS if a.name == target)
                    gaze_target = (cx + rng.uniform(-a.w / 3, a.w / 3), cy + rng.uniform(-a.h / 3, a.h / 3))
                    next_saccade = t + (rng.uniform(0.15, 0.3) if saccade_burst else rng.uniform(0.4, 1.2))
                    is_sacc = True
                gaze += (np.array(gaze_target) - gaze) * 0.5
                self.fusion.add_gaze(GazeSample(t, float(gaze[0]), float(gaze[1]), is_sacc))
                if frame % int(self.fps / 2) == 0:
                    g, n, tr = ph.rula
                    self.fusion.add_rula(RulaSample(t, g, n, tr))
                if ph.friction_rate and rng.random() < 0.15 * dt:
                    self.fusion.add_task_completion(t)

                # pupil stream, as an eye tracker would send it: baseline plus load-driven dilation, invalid in blinks
                self.fusion.add_pupil(PupilSample(t, 3.5 + ph.pupil_gain_mm * au4_slow + rng.normal(0, 0.02),
                                                  valid=t >= blink_until, luminance=0.5))
                occluded = occl_start is not None and occl_start <= t < occl_start + ph.occlusion_s
                lm = None if occluded else render(st, W, H, noise_px=0.25, rng=rng)
                obs = FaceObservation(t, lm, W, H, frame, self.start_utc_ms + int(t * 1000))
                payload = self.engine.process(obs)
                yield payload
                t += dt
                frame += 1


def run_realtime(sim: ScenarioSimulator, sinks: list, speed: float = 1.0) -> None:
    """Play a scenario at wall-clock pace (scaled by ``speed``) into sinks with .publish()."""
    t0 = time.perf_counter()
    for i, payload in enumerate(sim.run()):
        target = t0 + (i / sim.fps) / speed
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        for s in sinks:
            s.publish(payload)
