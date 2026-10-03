"""AU24 (lips pressed together): detection, and the look-alikes it must not fire on."""

import math

import numpy as np
import pytest

from cogsense.au_engine import AURegressor, geometric_evidence, lip_press_gate
from cogsense.calibration import Baseline
from cogsense.config import CogSenseConfig
from cogsense.features import FEATURE_NAMES, extract
from cogsense.geometry import normalize, to_pixel_space
from cogsense.synthetic import FaceState, render

KEY = "au24_lip_presser"
POSES = [(0, 0, 0), (28, 0, 0), (-28, 10, 0), (0, 18, -18), (20, -18, 15)]


def test_lip_thickness_is_a_feature_and_thins_when_pressed():
    thick = lambda st: extract(normalize(to_pixel_space(render(st), 1920, 1080)).points)["lip_thickness"]
    rest, pressed = thick(FaceState()), thick(FaceState(au24=1.0))
    assert "lip_thickness" in FEATURE_NAMES
    assert 1 - pressed / rest == pytest.approx(0.40, abs=0.01)  # 1.0 intensity = the documented 40 % thinning


@pytest.mark.parametrize("pose", POSES)
@pytest.mark.parametrize("level", [0.4, 0.8])
def test_press_recovered_across_pose(driver, pose, level):
    yaw, pitch, roll = pose
    p = driver.hold(0.5, state=FaceState(au24=level, yaw=yaw, pitch=pitch, roll=roll))
    assert p["tracking_status"] == "LOCKED"
    assert p["action_units"][KEY] == pytest.approx(level, abs=0.12)


def test_intensity_monotonic_and_no_leak_into_other_aus(driver):
    vals = [driver.hold(0.3, au24=a)["action_units"][KEY] for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals == sorted(vals) and vals[0] < 0.05 and vals[-1] > 0.9
    p = driver.hold(0.4, au24=0.8)
    for k in ("au04_brow_lowerer", "au07_lid_tightener", "au01_inner_brow_raiser", "au02_outer_brow_raiser", "au14_dimpler"):
        assert p["action_units"][k] < 0.1, k


def test_neutral_face_reads_zero(driver):
    assert driver.hold(1.0)["action_units"][KEY] < 0.03


def test_closed_lip_smile_is_not_a_lip_press(driver):
    """A closed-lip smile also thins the lips, but it widens the mouth, which discounts AU24."""
    p = driver.hold(0.6, smile=1.0)
    assert p["action_units"][KEY] < 0.15


def test_open_mouth_is_not_a_lip_press(driver):
    p = driver.hold(0.6, mouth_open=1.0)  # jaw drop
    assert p["action_units"][KEY] < 0.05


def test_speech_attenuates_lip_press():
    cfg = CogSenseConfig(blendshape_weight=0.0)
    feats = extract(normalize(to_pixel_space(render(FaceState(au24=0.8)), 1920, 1080)).points)
    rest = extract(normalize(to_pixel_space(render(FaceState()), 1920, 1080)).points)
    base = Baseline({k: rest[k] for k in FEATURE_NAMES})
    quiet = AURegressor(cfg).update(0.0, feats, base, None, False, False).au24
    talking = AURegressor(cfg).update(0.0, feats, base, None, False, True).au24
    assert quiet > 0.7
    assert talking == pytest.approx(quiet * cfg.speech_lower_face_attenuation, rel=0.02)


def test_blendshape_is_baseline_relative_and_gated_by_the_mouth():
    cfg = CogSenseConfig(blendshape_weight=0.5)
    neutral = extract(normalize(to_pixel_space(render(FaceState()), 1920, 1080)).points)
    # on a real face mouthPress already reads 0.10-0.16 at rest, so the resting value must be subtracted
    base = Baseline({k: neutral[k] for k in FEATURE_NAMES}, blendshapes={"mouthPressLeft": 0.15, "mouthPressRight": 0.15})
    resting = {"mouthPressLeft": 0.15, "mouthPressRight": 0.15}
    pressing = {"mouthPressLeft": 0.85, "mouthPressRight": 0.85}
    r = lambda feats, bs: AURegressor(cfg).update(0.0, feats, base, bs, False, False).au24
    assert r(neutral, resting) < 0.02  # resting blendshape alone is not a press
    both = extract(normalize(to_pixel_space(render(FaceState(au24=0.8)), 1920, 1080)).points)
    assert r(both, pressing) > r(both, None) * 0.9  # geometry + blendshape agree
    assert r(neutral, pressing) == pytest.approx(0.5 * 0.82, abs=0.05)  # blendshape alone counts at its fusion weight
    opened = extract(normalize(to_pixel_space(render(FaceState(mouth_open=1.0)), 1920, 1080)).points)
    assert r(opened, pressing) < 0.02  # ...but never with the mouth open


def test_gate_function_bounds():
    cfg = CogSenseConfig()
    neutral = extract(normalize(to_pixel_space(render(FaceState()), 1920, 1080)).points)
    base = Baseline({k: neutral[k] for k in FEATURE_NAMES})
    assert lip_press_gate(neutral, base, cfg) == pytest.approx(1.0)
    wide = dict(neutral, mouth_width=neutral["mouth_width"] + 0.2)
    assert lip_press_gate(wide, base, cfg) == 0.0
    open_ = dict(neutral, mouth_aperture=neutral["mouth_aperture"] + 0.1)
    assert lip_press_gate(open_, base, cfg) == 0.0
    jitter = dict(neutral, mouth_aperture=neutral["mouth_aperture"] + 0.008)  # inside the dead-zone: noise, not an opening
    assert lip_press_gate(jitter, base, cfg) == pytest.approx(1.0)


def test_baseline_saved_before_this_feature_still_works():
    cfg = CogSenseConfig()
    feats = extract(normalize(to_pixel_space(render(FaceState(au24=1.0)), 1920, 1080)).points)
    old = Baseline({k: feats[k] for k in FEATURE_NAMES if k != "lip_thickness"})  # an older baseline file
    assert geometric_evidence(feats, old, cfg)["au24"] == 0.0  # no crash, and no evidence without a rest value


def test_calibration_stores_resting_lip_thickness_and_press_blendshapes():
    from conftest import Driver

    d = Driver(fps=30)
    d.engine.start_calibration(0.0)
    bs = {"mouthPressLeft": 0.12, "mouthPressRight": 0.14}
    while d.engine.calibration_status == "CALIBRATING" and d.t < 20:
        d.engine.process(__import__("cogsense.engine", fromlist=["FaceObservation"]).FaceObservation(
            d.t, render(FaceState()), 1920, 1080, d.frame, blendshapes=bs))
        d.t += 1 / 30
        d.frame += 1
    base = d.engine.baseline
    assert base.features["lip_thickness"] == pytest.approx(0.3065, abs=0.01)
    assert base.blendshapes["mouthPressLeft"] == pytest.approx(0.12)
    # after calibrating on a resting mouth, a press registers
    p = d.hold(0.5, au24=0.8)
    assert p["action_units"][KEY] > 0.6
