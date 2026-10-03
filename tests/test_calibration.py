import json
import math

import pytest

from cogsense.calibration import Baseline
from cogsense.synthetic import FaceState

from conftest import Driver

# Subject with a naturally low, asymmetric brow and a habitual squint.
MORPH = {107: (0, 0.05, 0), 55: (0, 0.05, 0), 336: (0, 0.02, 0), 285: (0, 0.02, 0), 145: (0, -0.02, 0), 374: (0, -0.02, 0)}


def test_neutral_calibration_absorbs_morphology():
    d = Driver(fps=30)
    d.engine.start_calibration(0.0)
    p = None
    t_blink = 0.0
    while d.engine.calibration_status == "CALIBRATING":
        blink = 1.0 if (d.t - t_blink) % 3.5 < 0.15 else 0.0
        p = d.step(state=FaceState(asym=MORPH, blink=blink))
        assert d.t < 20, "calibration did not finish"
    assert p["calibration"]["status"] == "CALIBRATED"
    assert d.engine.baseline.blink_rate_per_min == pytest.approx(60 / 3.5, rel=0.25)
    p = d.hold(0.5, state=FaceState(asym=MORPH))
    assert p["action_units"]["au04_brow_lowerer"] < 0.05
    assert p["action_units"]["au07_lid_tightener"] < 0.05
    p = d.hold(0.5, state=FaceState(asym=MORPH, au4=0.6))
    assert p["action_units"]["au04_brow_lowerer"] == pytest.approx(0.6, abs=0.1)


def test_calibration_takes_15_seconds():
    d = Driver(fps=30)
    d.engine.start_calibration(0.0)
    d.hold(14.0)
    assert d.engine.calibration_status == "CALIBRATING"
    d.hold(1.5)
    assert d.engine.calibration_status == "CALIBRATED"


def test_speech_frames_excluded_from_baseline():
    d = Driver(fps=30)
    d.engine.start_calibration(0.0)
    while d.engine.calibration_status == "CALIBRATING" and d.t < 25:
        talking = 5 < d.t < 9
        d.step(mouth_open=(0.5 + 0.5 * math.sin(d.t * 25)) if talking else 0.0)
    assert d.engine.baseline.features["mouth_aperture"] < 0.06


def test_baseline_round_trip(tmp_path):
    b = Baseline({"a": 1.0}, {"a": 0.1}, {"browDownLeft": 0.05}, 15.0, 400)
    path = tmp_path / "b.json"
    b.save(path)
    assert Baseline.load(path).to_dict() == b.to_dict()
    json.loads(path.read_text())
