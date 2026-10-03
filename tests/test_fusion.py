from cogsense.config import CogSenseConfig
from cogsense.fusion import AOI, CompoundRiskTracker, FusionHub, GazeSample, RulaSample, correlate

from conftest import Driver


def test_aoi_hit_test_and_staleness():
    hub = FusionHub(CogSenseConfig())
    hub.set_aois([AOI("A", 0, 0, 100, 100), AOI("B", 50, 50, 100, 100)])
    hub.add_gaze(GazeSample(1.0, 75, 75))
    assert hub.context(1.05).active_aoi == "B"  # top-most wins on overlap
    assert hub.context(2.0).active_aoi is None  # stale gaze is not used
    hub.add_gaze(GazeSample(2.0, 10, 10))
    assert hub.context(2.01).active_aoi == "A"


def test_nearest_sample_with_out_of_order_arrival():
    hub = FusionHub(CogSenseConfig())
    hub.add_rula(RulaSample(1.0, 3, 2, 2))
    hub.add_rula(RulaSample(3.0, 6, 4, 3))
    hub.add_rula(RulaSample(2.0, 5, 3, 3))
    assert hub.context(2.1).rula_grand_score == 5
    assert hub.context(10.0).rula_grand_score is None


def test_saccade_rate():
    hub = FusionHub(CogSenseConfig())
    for i in range(10):
        hub.add_gaze(GazeSample(i * 0.1, 0, 0, saccade=i % 2 == 0))
    assert hub.context(0.95).saccade_rate_hz == 5


def test_compound_posture_risk_edge_triggered():
    d = Driver()
    hub = d.engine.fusion
    d.warmup()
    n = 0
    for _ in range(int(3 * d.fps)):
        hub.add_rula(RulaSample(d.t, 6, 3, 2))
        p = d.step(au7=0.6)
        n += sum(1 for e in p["events"] if e["type"] == "COMPOUND_POSTURE_RISK")
    assert n == 1
    assert p["fusion_context"]["correlated_insight"] == "POSTURE_DRIVEN_VISUAL_COMPENSATION"
    assert p["fusion_context"]["rula_neck_score"] == 3


def test_tracker_rearms_after_clear():
    tr = CompoundRiskTracker(rearm_s=1.0)
    assert tr.update(0.0, True)
    assert not tr.update(0.1, True)
    assert not tr.update(0.2, False)
    assert not tr.update(0.5, True)  # brief dip does not re-trigger
    tr.update(0.6, False)
    tr.update(1.7, False)
    assert tr.update(1.8, True)


def test_insights():
    cfg = CogSenseConfig()
    from cogsense.fusion import FusionContext

    ctx = FusionContext("RADAR", 1, 1, 0, 3, 1, 1, None, None)
    assert correlate(ctx, 0.8, 0.0, 0.8, 50, False, False, cfg)[0] == "HIGH_COGNITIVE_STRAIN_ON_COMPLEX_WIDGET"
    assert correlate(ctx, 0.0, 0.6, 0.1, 10, False, False, cfg)[0] == "VISUAL_STRAIN_DISPLAY_LEGIBILITY"
    assert correlate(ctx, 0.0, 0.0, 0.1, 10, True, False, cfg)[0] == "AUTOMATION_SURPRISE"
