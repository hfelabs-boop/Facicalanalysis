import pytest

from cogsense.config import CogSenseConfig
from cogsense.engine import CogSenseEngine

from conftest import Driver


def run(d, seconds, closed_fn=None, face=True, **kw):
    """Step the driver; ``closed_fn(t)`` returns the blink value (0 open … 1 closed) for each frame."""
    p = None
    for _ in range(int(seconds * d.fps)):
        blink = closed_fn(d.t) if closed_fn else 0.0
        p = d.step(face=face, blink=blink, **kw)
    return p


def periodic_closure(period, closed_s, start=0.0):
    return lambda t: 1.0 if t >= start and ((t - start) % period) < closed_s else 0.0


@pytest.fixture
def d30():
    d = Driver(fps=30)
    d.warmup()
    return d


def test_perclos_matches_closed_fraction(d30):
    # eyes fully closed 0.6 s out of every 3 s = 20 %
    p = run(d30, 70, periodic_closure(3.0, 0.6, start=d30.t))
    assert p["fatigue"]["perclos"] == pytest.approx(0.20, abs=0.04)
    assert p["fatigue"]["window_s"] == 60.0
    assert p["fatigue"]["coverage"] > 0.95


def test_partial_closure_is_not_counted_as_closed(d30):
    p = run(d30, 40, lambda t: 0.5)  # eyelids half down all the time: squinting, not closed
    assert p["fatigue"]["perclos"] == 0.0


def test_no_value_until_enough_valid_data(d30):
    p = run(d30, 5)
    assert p["fatigue"]["perclos"] is None
    p = run(d30, 8)
    assert p["fatigue"]["perclos"] is not None


def test_tracking_gaps_are_not_credited_as_open_or_closed_time(d30):
    run(d30, 20)  # eyes open, 20 s valid
    run(d30, 30, face=False)  # 30 s with no face
    p = run(d30, 10, lambda t: 1.0)  # then 10 s fully closed
    f = p["fatigue"]
    # Only valid frames count: ~10 s closed of ~30 s valid, never closed/60 s or diluted by the gap
    assert f["perclos"] == pytest.approx(10 / 30, abs=0.05)
    assert 0.4 < f["coverage"] < 0.6  # about half the elapsed window had usable tracking


def test_window_slides(d30):
    run(d30, 30, lambda t: 1.0)
    p = run(d30, 70)  # open for longer than the window
    assert p["fatigue"]["perclos"] == 0.0


def test_blink_duration_and_long_closures(d30):
    t0 = d30.t
    # three short blinks (0.2 s) and two long closures (0.8 s), well separated
    starts = [(2, 0.2), (6, 0.2), (10, 0.8), (16, 0.2), (21, 0.8)]
    def closed(t):
        return 1.0 if any(t0 + s <= t < t0 + s + L for s, L in starts) else 0.0
    p = run(d30, 30, closed)
    f = p["fatigue"]
    assert f["blink_count"] == 5
    assert f["long_closures"] == 2
    assert f["mean_blink_ms"] == pytest.approx((3 * 200 + 2 * 800) / 5, rel=0.15)


def test_no_alert_threshold_by_default(d30):
    p = run(d30, 70, lambda t: 1.0 if (t % 2) < 1.0 else 0.0)
    assert p["fatigue"]["perclos"] > 0.4
    assert not any(e["type"] == "FATIGUE_PERCLOS" for e in p["events"])


def test_alert_when_configured_fires_once_per_window():
    d = Driver(engine=CogSenseEngine(CogSenseConfig(perclos_alert=0.15)), fps=30)
    d.warmup()
    events = []
    for _ in range(int(80 * d.fps)):
        p = d.step(blink=1.0 if (d.t % 2) < 0.6 else 0.0)
        events += [e for e in p["events"] if e["type"] == "FATIGUE_PERCLOS"]
    assert 1 <= len(events) <= 2
    assert events[0]["perclos"] >= 0.15


def test_lost_tracking_reports_no_fatigue_values(d30):
    run(d30, 15)
    p = d30.step(face=False)
    assert p["fatigue"] is None


def test_isolated_frames_after_dropouts_are_never_credited():
    """A frame that reappears after a dropout has no valid predecessor, so it adds no time at all.

    Without that rule each reappearance would credit up to the frame-gap cap as closed time, and a
    flickering tracker with eyes shut on every reappearance would read as PERCLOS 1.0.
    """
    cfg = CogSenseConfig(perclos_min_coverage_s=2.0)
    d = Driver(engine=CogSenseEngine(cfg), fps=30)
    d.warmup()
    p = None
    for _ in range(120):
        for _ in range(30):
            d.step(face=False)
        p = d.step(blink=1.0)  # one closed frame, then the face vanishes again
    assert p["fatigue"]["perclos"] is None  # no valid duration accumulated, so nothing is reported
