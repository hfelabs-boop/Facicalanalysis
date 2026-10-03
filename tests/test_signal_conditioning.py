import math

from cogsense.synthetic import FaceState


def test_speech_is_flagged_and_attenuates_lower_face(driver):
    quiet = driver.hold(0.5, au14=0.6)["action_units"]["au14_dimpler"]
    p = None
    for _ in range(60):
        p = driver.step(au14=0.6, mouth_open=0.5 + 0.5 * math.sin(driver.t * 2 * math.pi * 4))
    assert p["cognitive_metrics"]["speech_interference_detected"] is True
    assert p["action_units"]["au14_dimpler"] < 0.5 * quiet
    assert p["confidence"] < 1.0


def test_speech_does_not_bias_upper_face(driver):
    p = None
    for _ in range(60):
        p = driver.step(mouth_open=0.5 + 0.5 * math.sin(driver.t * 2 * math.pi * 4))
    for k in ("au04_brow_lowerer", "au01_inner_brow_raiser", "au02_outer_brow_raiser", "au07_lid_tightener"):
        assert p["action_units"][k] < 0.05


def test_speech_flag_clears(driver):
    for _ in range(60):
        driver.step(mouth_open=0.5 + 0.5 * math.sin(driver.t * 25))
    assert driver.hold(1.5)["cognitive_metrics"]["speech_interference_detected"] is False


def test_no_face_reports_lost_without_stale_metrics(driver):
    driver.hold(0.5, au4=0.9)
    p = driver.step(face=False)
    assert p["tracking_status"] == "LOST"
    assert p["confidence"] < 0.3
    assert p["action_units"] is None and p["cognitive_metrics"] is None


def test_extreme_pose_drops_confidence(driver):
    p = driver.hold(0.3, state=FaceState(yaw=60))
    assert p["confidence"] < 0.3
    assert p["action_units"] is None
    ok = driver.hold(0.3, state=FaceState(yaw=28))
    assert ok["tracking_status"] == "LOCKED"
    assert ok["head_pose"]["within_operating_range"] is True


def test_tracker_glitch_drops_confidence(driver):
    import numpy as np
    from cogsense.engine import FaceObservation
    from cogsense.synthetic import render

    lm = render(FaceState())
    lm[:200] += np.random.default_rng(0).normal(0, 0.03, (200, 3))
    p = driver.engine.process(FaceObservation(driver.t, lm, 1920, 1080, 999))
    assert p["confidence"] < 0.6


def test_recovery_after_loss(driver):
    driver.hold(0.5, face=False)
    p = driver.hold(0.3, au4=0.7)
    assert p["tracking_status"] == "LOCKED"
    assert p["action_units"]["au04_brow_lowerer"] > 0.6
