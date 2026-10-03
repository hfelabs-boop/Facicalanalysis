import pytest

from cogsense.fusion import AOI, GazeSample
from cogsense.synthetic import FaceState

from conftest import Driver


def events(p, kind):
    return [e for e in p["events"] if e["type"] == kind]


def run_collect(d, seconds, **kw):
    out = []
    for _ in range(int(seconds * d.fps)):
        out.append(d.step(**kw))
    return out


def blinking(d, seconds, au4, period=60 / 17):
    """Hold an AU4 level while blinking at the resting rate (so blink suppression stays ~0)."""
    p = None
    for _ in range(int(seconds * d.fps)):
        p = d.step(au4=au4, blink=1.0 if (d.t % period) < 0.15 else 0.0)
    return p["cognitive_metrics"]["mental_effort_score"]


def test_mes_rises_with_sustained_au4(driver):
    blinking(driver, 20, 0.0)
    low = blinking(driver, 6, 0.0)
    high = blinking(driver, 6, 0.8)
    assert low < 8
    assert 50 < high <= 100


def test_mes_uses_five_second_window(driver):
    blinking(driver, 20, 0.0)
    high = blinking(driver, 6, 0.8)
    mid = blinking(driver, 2.5, 0.0)
    assert 0.3 * high < mid < 0.7 * high  # half the window still elevated
    assert blinking(driver, 3.0, 0.0) < 8


def test_blink_suppression_increases_mes():
    def session(blink_period):
        d = Driver(fps=30)
        d.warmup()
        p = None
        for i in range(int(40 * 30)):
            closed = blink_period and (d.t % blink_period) < 0.15
            p = d.step(au4=0.4, blink=1.0 if closed else 0.0)
        return p["cognitive_metrics"]["mental_effort_score"]

    normal = session(60 / 17)  # resting blink rate
    suppressed = session(0)  # no blinks
    assert suppressed > normal + 15


def spike(d, au14=0.0, ramp_frames=6, hold_s=0.5):
    out = []
    for i in range(ramp_frames):
        out.append(d.step(au4=0.9 * (i + 1) / ramp_frames, au14=au14))
    out += run_collect(d, hold_s, au4=0.9, au14=au14)
    return out


def test_cfi_event_on_spike_with_au14(driver):
    frames = spike(driver, au14=0.6)
    evs = [e for p in frames for e in events(p, "CFI_EVENT")]
    assert len(evs) == 1
    assert evs[0]["trigger"] == "AU14"
    assert max(p["cognitive_metrics"]["cognitive_friction_index"] for p in frames) > 0.7


def test_no_cfi_event_without_co_occurrence(driver):
    frames = spike(driver, au14=0.0)
    assert not [e for p in frames for e in events(p, "CFI_EVENT")]


def test_no_cfi_event_for_slow_rise(driver):
    frames = []
    for i in range(120):  # 2 s ramp: sustained effort, not a spike
        frames.append(driver.step(au4=0.9 * i / 119, au14=0.6))
    assert not [e for p in frames for e in events(p, "CFI_EVENT")]


def test_cfi_event_on_spike_with_saccades():
    d = Driver()
    hub = d.engine.fusion
    hub.set_aois([AOI("RADAR", 0, 0, 1000, 1000)])
    d.warmup()
    frames = []
    for i in range(40):
        hub.add_gaze(GazeSample(d.t, 500 + (i % 2) * 50, 500, saccade=(i % 3 == 0)))
        frames.append(d.step(au4=min(0.9, 0.15 * (i + 1))))
    evs = [e for p in frames for e in events(p, "CFI_EVENT")]
    assert evs and evs[0]["trigger"] == "SACCADES" and evs[0]["aoi"] == "RADAR"


def test_task_completion_suppresses_cfi_event(driver):
    driver.engine.fusion.add_task_completion(driver.t)
    frames = spike(driver, au14=0.6)
    assert not [e for p in frames for e in events(p, "CFI_EVENT")]


def surprise(d, gap_s):
    frames = run_collect(d, 0.3, au1=0.8, au2=0.8)
    frames += run_collect(d, gap_s)
    for i in range(6):
        frames.append(d.step(au4=0.8 * (i + 1) / 6))
    frames += run_collect(d, 0.3, au4=0.8)
    return frames


def test_automation_surprise_within_800ms(driver):
    frames = surprise(driver, 0.3)
    assert sum(len(events(p, "AUTOMATION_SURPRISE")) for p in frames) == 1
    assert any(p["cognitive_metrics"]["automation_surprise_flag"] for p in frames)


def test_no_automation_surprise_after_800ms(driver):
    frames = surprise(driver, 1.2)
    assert not any(events(p, "AUTOMATION_SURPRISE") for p in frames)


def test_surprise_flag_latches_then_clears(driver):
    surprise(driver, 0.2)
    assert driver.hold(1.5)["cognitive_metrics"]["automation_surprise_flag"] is False
