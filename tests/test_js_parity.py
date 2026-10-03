"""The browser/Node engine (dashboard/engine.js) must agree with the Python engine frame by frame."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cogsense.engine import CogSenseEngine, FaceObservation
from cogsense.synthetic import FaceState, render

ROOT = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

FPS = 30


def scenario():
    """(FaceState | None, blendshapes | None) per frame covering the engine's branches."""
    frames = []

    def hold(seconds, fn):
        for i in range(int(seconds * FPS)):
            t = len(frames) / FPS
            frames.append(fn(t, i))

    hold(1.0, lambda t, i: (FaceState(), {"browDownLeft": 0.3, "browDownRight": 0.3, "mouthDimpleLeft": 0.02}))
    # 15 s calibration: blinks every 3.5 s, talking 6–8 s, slight natural asymmetry
    morph = {107: (0, 0.03, 0), 55: (0, 0.03, 0)}

    def calib(t, i):
        st = FaceState(asym=morph, blink=1.0 if (t % 3.5) < 0.15 else 0.0)
        if 6 < t < 8:
            st.mouth_open = 0.5 + 0.5 * math.sin(t * 25)
        return st, {"browDownLeft": 0.05, "browDownRight": 0.05}

    hold(16.5, calib)
    hold(1.0, lambda t, i: (FaceState(asym=morph), {"browDownLeft": 0.05, "browDownRight": 0.05}))
    # AU4 + AU14 spike (CFI event)
    hold(0.3, lambda t, i: (FaceState(asym=morph, au4=0.9 * (i + 1) / 9, au14=0.6), {"browDownLeft": 0.8, "browDownRight": 0.8}))
    hold(0.7, lambda t, i: (FaceState(asym=morph, au4=0.9, au14=0.6), {"browDownLeft": 0.8, "browDownRight": 0.8}))
    hold(1.5, lambda t, i: (FaceState(asym=morph), None))
    # AU1+2 then AU4 surge (automation surprise)
    hold(0.3, lambda t, i: (FaceState(asym=morph, au1=0.8, au2=0.8), None))
    hold(0.3, lambda t, i: (FaceState(asym=morph), None))
    hold(0.2, lambda t, i: (FaceState(asym=morph, au4=0.8 * (i + 1) / 6), None))
    hold(0.5, lambda t, i: (FaceState(asym=morph, au4=0.8), None))
    hold(1.5, lambda t, i: (FaceState(asym=morph), None))
    hold(1.0, lambda t, i: (FaceState(asym=morph, au7=0.7, yaw=22, pitch=-12, roll=8), None))  # pose + squint
    hold(0.7, lambda t, i: (FaceState(asym=morph, blink=1.0), None))  # microsleep
    hold(0.5, lambda t, i: (FaceState(asym=morph), None))  # eyes reopen: the long closure completes
    hold(0.5, lambda t, i: (None, None))  # no face
    hold(0.5, lambda t, i: (FaceState(asym=morph, yaw=60), None))  # beyond pose range
    hold(1.0, lambda t, i: (FaceState(asym=morph, au14=0.6, mouth_open=0.5 + 0.5 * math.sin(t * 25)), None))  # speech
    hold(1.0, lambda t, i: (FaceState(asym=morph), None))
    # lips pressed together (with and without the blendshape), a closed-lip smile, and a press while talking
    press_bs = {"mouthPressLeft": 0.7, "mouthPressRight": 0.7}
    hold(1.0, lambda t, i: (FaceState(asym=morph, au24=0.8), press_bs))
    hold(0.7, lambda t, i: (FaceState(asym=morph, au24=0.5), None))
    hold(0.8, lambda t, i: (FaceState(asym=morph, smile=1.0), {"mouthPressLeft": 0.2, "mouthPressRight": 0.2}))
    hold(1.0, lambda t, i: (FaceState(asym=morph), None))
    hold(1.0, lambda t, i: (FaceState(asym=morph, au24=0.8, mouth_open=0.5 + 0.5 * math.sin(t * 25)), None))
    hold(1.0, lambda t, i: (FaceState(asym=morph), None))
    return frames


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("parity")
    specs = scenario()
    frames, py_out = [], []
    eng = CogSenseEngine()
    calibrate_at = FPS  # 1.0 s
    for i, (st, bs) in enumerate(specs):
        t = i / FPS
        lm = None if st is None else np.round(render(st), 6)
        utc = 1_700_000_000_000 + int(t * 1000)
        frames.append({"t": t, "lm": None if lm is None else lm.tolist(), "w": 1920, "h": 1080, "id": i, "utc": utc, "bs": bs})
        if i == calibrate_at:
            eng.start_calibration(t)
        py_out.append(eng.process(FaceObservation(t, lm, 1920, 1080, i, utc, bs)))
    (d / "in.json").write_text(json.dumps({"frames": frames, "calibrate_at": calibrate_at}))
    subprocess.run(["node", str(ROOT / "js_run.js"), str(d / "in.json"), str(d / "out.json")], check=True, timeout=120)
    return py_out, json.loads((d / "out.json").read_text())


def test_same_length_and_status(runs):
    py, js = runs
    assert len(py) == len(js)
    assert [p["tracking_status"] for p in py] == [p["tracking_status"] for p in js]
    assert [p["calibration"]["status"] for p in py] == [p["calibration"]["status"] for p in js]


def test_values_match(runs):
    py, js = runs
    worst = 0.0
    for i, (a, b) in enumerate(zip(py, js)):
        assert abs(a["confidence"] - b["confidence"]) < 2e-3, i
        if a["action_units"] is None:
            assert b["action_units"] is None and b["cognitive_metrics"] is None, i
            continue
        for k, v in a["action_units"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                worst = max(worst, abs(v - b["action_units"][k]))
                assert abs(v - b["action_units"][k]) < 2e-3, (i, k, v, b["action_units"][k])
            else:
                assert v == b["action_units"][k], (i, k)
        for k in ("mental_effort_score", "cognitive_friction_index"):
            assert abs(a["cognitive_metrics"][k] - b["cognitive_metrics"][k]) < 0.05, (i, k)
        for k in ("automation_surprise_flag", "speech_interference_detected"):
            assert a["cognitive_metrics"][k] == b["cognitive_metrics"][k], (i, k)
        for k in ("yaw_deg", "pitch_deg", "roll_deg"):
            assert abs(a["head_pose"][k] - b["head_pose"][k]) < 0.01, (i, k)
        assert a["fusion_context"]["correlated_insight"] == b["fusion_context"]["correlated_insight"], i
        fa, fb = a["fatigue"], b["fatigue"]
        assert (fa["perclos"] is None) == (fb["perclos"] is None), (i, fa, fb)
        if fa["perclos"] is not None:
            assert abs(fa["perclos"] - fb["perclos"]) < 2e-3, (i, fa, fb)
        assert abs(fa["coverage"] - fb["coverage"]) < 2e-3, i
        assert fa["blink_count"] == fb["blink_count"] and fa["long_closures"] == fb["long_closures"], (i, fa, fb)
        assert (fa["mean_blink_ms"] is None) == (fb["mean_blink_ms"] is None), i
        if fa["mean_blink_ms"] is not None:
            assert abs(fa["mean_blink_ms"] - fb["mean_blink_ms"]) < 0.2, (i, fa, fb)
    assert worst < 2e-3


def test_events_match_and_scenario_exercises_them(runs):
    py, js = runs
    ev = lambda out: [(i, e["type"]) for i, p in enumerate(out) for e in p["events"]]
    assert ev(py) == ev(js)
    kinds = {k for _, k in ev(py)}
    assert {"CFI_EVENT", "AUTOMATION_SURPRISE", "MICROSLEEP"} <= kinds
    assert any(p["cognitive_metrics"] and p["cognitive_metrics"]["speech_interference_detected"] for p in py)
    assert py[-1]["calibration"]["status"] == "CALIBRATED"
    assert max(p["action_units"]["au24_lip_presser"] for p in py if p["action_units"]) > 0.6  # AU24 path exercised
    last = [p for p in py if p["fatigue"]][-1]["fatigue"]
    assert last["perclos"] is not None and last["blink_count"] > 0 and last["long_closures"] >= 1  # fatigue path exercised


def test_calibrated_baseline_agrees(runs):
    py, js = runs
    a, b = py[-1]["blink"], js[-1]["blink"]
    assert a["rate_per_min"] == pytest.approx(b["rate_per_min"], abs=0.2)
