import pytest

from cogsense.synthetic import FaceState

AU_FIELDS = {
    "au4": "au04_brow_lowerer",
    "au7": "au07_lid_tightener",
    "au1": "au01_inner_brow_raiser",
    "au2": "au02_outer_brow_raiser",
    "au14": "au14_dimpler",
}
POSES = [(0, 0, 0), (30, 0, 0), (-30, 10, 0), (0, 20, -20), (20, -20, 20)]


@pytest.mark.parametrize("au", list(AU_FIELDS))
@pytest.mark.parametrize("pose", POSES)
def test_isolated_au_recovered_across_pose(driver, au, pose):
    yaw, pitch, roll = pose
    p = driver.hold(0.5, state=FaceState(**{au: 0.8}, yaw=yaw, pitch=pitch, roll=roll))
    aus = p["action_units"]
    assert p["tracking_status"] == "LOCKED"
    assert aus[AU_FIELDS[au]] == pytest.approx(0.8, abs=0.12)
    for other, field in AU_FIELDS.items():
        if other != au:
            assert aus[field] < 0.1, f"{field} leaked from {au}: {aus[field]}"


def test_intensity_is_monotonic(driver):
    vals = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        vals.append(driver.hold(0.3, au4=a)["action_units"]["au04_brow_lowerer"])
    assert vals == sorted(vals)
    assert vals[0] < 0.05 and vals[-1] > 0.9


def test_outputs_are_normalized_floats(driver):
    p = driver.hold(0.3, au4=1.0, au7=1.0, au1=1.0, au2=1.0, au14=1.0)
    for field in AU_FIELDS.values():
        assert 0.0 <= p["action_units"][field] <= 1.0


def test_blink_is_not_reported_as_lid_tightening(driver):
    p = None
    for _ in range(12):
        p = driver.step(blink=1.0)
        assert p["action_units"]["au07_lid_tightener"] < 0.1
    assert p["action_units"]["au45_blink_state"] == 1
    assert driver.hold(0.2)["action_units"]["au45_blink_state"] == 0


def test_microsleep_sets_au43(driver):
    p = driver.hold(0.7, blink=1.0)
    assert p["action_units"]["au43_eyes_closed"] is True
    assert p["fusion_context"]["correlated_insight"] == "FATIGUE_MICROSLEEP_RISK"


def test_transient_spike_under_500ms_is_captured(driver):
    peak = 0.0
    for i in range(18):  # 300 ms burst at 60 Hz
        peak = max(peak, driver.step(au4=0.9)["action_units"]["au04_brow_lowerer"])
    assert peak > 0.8


def test_blendshape_evidence_is_fused():
    from cogsense.config import CogSenseConfig
    from cogsense.engine import CogSenseEngine, FaceObservation
    from cogsense.synthetic import render

    eng = CogSenseEngine(CogSenseConfig(blendshape_weight=1.0))
    lm = render(FaceState())
    for i in range(40):
        p = eng.process(FaceObservation(i / 60, lm, 1920, 1080, i, blendshapes={"browDownLeft": 0.0, "browDownRight": 0.0}))
    for i in range(40, 70):
        p = eng.process(FaceObservation(i / 60, lm, 1920, 1080, i, blendshapes={"browDownLeft": 0.7, "browDownRight": 0.7}))
    assert p["action_units"]["au04_brow_lowerer"] == pytest.approx(0.7, abs=0.05)
