import math

import pytest

from cogsense.config import CogSenseConfig
from cogsense.fusion import FusionHub, PupilSample
from cogsense.pupil import PupilTracker

from conftest import Driver

HZ = 60


def feed(hub, tr, t0, seconds, diameter, luma=None, valid=True, calibrating=False):
    out = None
    for i in range(int(seconds * HZ)):
        t = t0 + i / HZ
        hub.add_pupil(PupilSample(t, diameter(t) if callable(diameter) else diameter, valid, luma))
        out = tr.update(t)
    return out, t0 + seconds


def make():
    cfg = CogSenseConfig()
    hub = FusionHub(cfg)
    return cfg, hub, PupilTracker(cfg, hub)


def test_no_data_means_no_reading():
    _, hub, tr = make()
    assert tr.update(1.0) is None


def test_stale_data_is_not_reported():
    _, hub, tr = make()
    hub.add_pupil(PupilSample(1.0, 3.5))
    assert tr.update(1.2) is not None
    assert tr.update(3.0) is None  # nothing within the smoothing window


def test_invalid_samples_are_ignored():
    _, hub, tr = make()
    hub.add_pupil(PupilSample(1.0, 3.5, valid=False))
    hub.add_pupil(PupilSample(1.01, float("nan")))
    hub.add_pupil(PupilSample(1.02, 0.0))
    hub.add_pupil(PupilSample(1.03, -1.0))
    assert tr.update(1.05) is None
    hub.add_pupil(PupilSample(1.04, 3.2))
    assert tr.update(1.05)["diameter_mm"] == 3.2


def test_provisional_baseline_then_change():
    cfg, hub, tr = make()
    r, t = feed(hub, tr, 0.0, 2.0, 3.5)
    assert r["baseline_mm"] is None and r["change_mm"] is None and r["reliable"] is False  # still learning
    r, t = feed(hub, tr, t, 6.0, 3.5)
    assert r["baseline_source"] == "provisional" and r["baseline_mm"] == pytest.approx(3.5)
    r, t = feed(hub, tr, t, 2.0, 3.9)
    assert r["change_mm"] == pytest.approx(0.4, abs=0.01)
    assert r["change_pct"] == pytest.approx(100 * 0.4 / 3.5, abs=0.3)
    assert r["reliable"] is True


def test_calibration_baseline_replaces_provisional_and_uses_median():
    cfg, hub, tr = make()
    r, t = feed(hub, tr, 0.0, 6.0, 3.0)  # provisional baseline 3.0
    tr.start_calibration()
    # calibration with a brief outlier (median must resist it)
    r, t = feed(hub, tr, t, 6.0, lambda x: 3.6 if 8.0 < x < 8.2 else 3.4)
    tr.finish_calibration()
    r, t = feed(hub, tr, t, 1.0, 3.8)
    assert r["baseline_source"] == "calibration"
    assert r["baseline_mm"] == pytest.approx(3.4, abs=0.02)
    assert r["change_mm"] == pytest.approx(0.4, abs=0.03)


def test_luminance_shift_flags_reading_unreliable():
    cfg, hub, tr = make()
    r, t = feed(hub, tr, 0.0, 7.0, 3.5, luma=0.50)
    r, t = feed(hub, tr, t, 1.0, 3.2, luma=0.52)  # +4 %: within tolerance
    assert r["reliable"] is True and r["luminance_shift"] == pytest.approx(0.04, abs=0.01)
    r, t = feed(hub, tr, t, 1.0, 3.0, luma=0.65)  # +30 %: the light changed, not the workload
    assert r["reliable"] is False and r["luminance_shift"] == pytest.approx(0.30, abs=0.02)


def test_without_luminance_input_reading_is_still_reported_but_unchecked():
    _, hub, tr = make()
    r, t = feed(hub, tr, 0.0, 8.0, 3.5)
    assert r["luminance_shift"] is None and r["reliable"] is True


def test_end_to_end_through_the_engine():
    d = Driver(fps=30)
    hub = d.engine.fusion
    d.warmup()
    p = None
    for i in range(int(8 * d.fps)):
        hub.add_pupil(PupilSample(d.t, 3.5, True, 0.5))
        p = d.step()
    pu = p["fusion_context"]["pupil"]
    assert pu["diameter_mm"] == pytest.approx(3.5, abs=0.01)
    assert pu["baseline_mm"] == pytest.approx(3.5, abs=0.01)
    # engine without pupil input reports none rather than a stale value
    d2 = Driver(fps=30)
    d2.warmup()
    assert d2.step()["fusion_context"]["pupil"] is None


def test_pupil_message_without_diameter_is_ignored():
    from cogsense.bus import InboundRouter
    from cogsense.clock import SyncClock

    cfg = CogSenseConfig()
    hub = FusionHub(cfg)
    InboundRouter(hub, SyncClock()).handle({"type": "pupil", "valid": True})
    assert hub.pupil_between(-1e9, 1e9) == []
