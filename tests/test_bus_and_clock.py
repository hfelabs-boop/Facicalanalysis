import asyncio
import json
import time

import pytest

from cogsense.bus import LSL_CHANNELS, InboundRouter, WebSocketBroadcaster, payload_to_lsl_sample
from cogsense.clock import SyncClock
from cogsense.config import CogSenseConfig
from cogsense.fusion import FusionHub

from conftest import Driver


def test_clock_round_trip_and_drift():
    c = SyncClock()
    t = c.now()
    assert abs(c.from_utc_ms(c.to_utc_ms(t)) - t) < 0.001
    assert abs(c.to_utc_ms(t) - time.time() * 1000) < 10
    assert abs(c.drift_ms()) < 10
    assert c.from_lsl(c.to_lsl(t)) == pytest.approx(t)


def test_clock_slews_instead_of_stepping():
    c = SyncClock(resync_interval_s=0.0, max_slew_ms=0.5)
    c._utc_offset += 0.004  # inject 4 ms error
    t1 = c.to_utc_ms(c.now())
    c.now()
    assert c.max_observed_drift_ms == pytest.approx(4, abs=0.5)
    for _ in range(20):
        c.now()
    assert abs(c.drift_ms()) < 1.0
    assert c.to_utc_ms(c.now()) >= t1 - 5


def test_router_applies_messages():
    cfg = CogSenseConfig()
    hub = FusionHub(cfg)
    clock = SyncClock()
    calls = []
    r = InboundRouter(hub, clock, lambda cmd, msg: calls.append(cmd))
    now_ms = clock.to_utc_ms(clock.now())
    r.handle({"type": "aoi_layout", "aois": [{"name": "MAP", "x": 0, "y": 0, "w": 10, "h": 10}]})
    r.handle({"type": "gaze", "x": 5, "y": 5, "t_utc_ms": now_ms})
    r.handle({"type": "rula", "grand": 6, "neck": 3, "trunk": 4, "t_utc_ms": now_ms})
    r.handle({"type": "ui_event", "event": "TASK_COMPLETE", "t_utc_ms": now_ms})
    r.handle({"type": "phase", "name": "INGRESS"})
    r.handle({"type": "control", "cmd": "calibrate"})
    ctx = hub.context(clock.from_utc_ms(now_ms))
    assert (ctx.active_aoi, ctx.rula_neck_score, ctx.mission_phase) == ("MAP", 3, "INGRESS")
    assert ctx.seconds_since_task_complete == pytest.approx(0, abs=0.01)
    assert calls == ["calibrate"]


def test_websocket_round_trip():
    import websockets

    hub = FusionHub(CogSenseConfig())
    clock = SyncClock()
    ws = WebSocketBroadcaster("127.0.0.1", 0, InboundRouter(hub, clock))
    ws.start()
    try:
        async def client():
            async with websockets.connect(f"ws://127.0.0.1:{ws.port}") as c:
                await c.send(json.dumps([{"type": "aoi_layout", "aois": [{"name": "W", "x": 0, "y": 0, "w": 5, "h": 5}]},
                                         {"type": "gaze", "x": 1, "y": 1}]))
                await asyncio.sleep(0.2)
                ws.publish({"frame_id": 7})
                return json.loads(await asyncio.wait_for(c.recv(), 2))

        msg = asyncio.run(client())
        assert msg == {"frame_id": 7}
        assert hub.context(clock.now()).active_aoi == "W"
    finally:
        ws.stop()


def test_lsl_sample_layout():
    d = Driver()
    d.warmup()
    p = d.hold(1.0, au4=0.5)
    s = payload_to_lsl_sample(p)
    assert len(s) == len(LSL_CHANNELS)
    assert s[LSL_CHANNELS.index("au04")] == pytest.approx(0.5, abs=0.1)
    lost = payload_to_lsl_sample(d.step(face=False))
    assert lost[LSL_CHANNELS.index("au04")] != lost[LSL_CHANNELS.index("au04")]  # NaN, not stale


def test_lsl_outlet_publishes():
    pylsl = pytest.importorskip("pylsl")
    from cogsense.bus import LSLPublisher

    try:
        pub = LSLPublisher(SyncClock(), 60.0, source_id="cogsense-test")
    except Exception as exc:  # pragma: no cover - liblsl/network environment
        pytest.skip(f"LSL unavailable: {exc}")
    d = Driver()
    p = d.hold(1.0)
    pub.publish(p, time.perf_counter())
