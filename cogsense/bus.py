"""Multimodal synchronization bus: WebSocket JSON broadcast, LSL outlets/inlets,
and JSONL telemetry logging (FR-4.1).

Inbound WebSocket messages (from the eye tracker, RULA service, C2 UI or
the dashboard) are JSON objects with a ``type`` field:

  {"type": "gaze", "x": 812, "y": 440, "saccade": false, "t_utc_ms": 1790947625101}
  {"type": "rula", "grand": 5, "neck": 3, "trunk": 2, "t_utc_ms": ...}
  {"type": "ui_event", "event": "TASK_COMPLETE", "t_utc_ms": ...}
  {"type": "aoi_layout", "aois": [{"name": "RADAR", "x": 0, "y": 0, "w": 800, "h": 600}]}
  {"type": "phase", "name": "SWARM_ENGAGEMENT"}
  {"type": "control", "cmd": "calibrate" | "record_start" | "record_stop"}

``t_utc_ms`` is optional; samples without it are stamped on arrival.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Callable

from cogsense.clock import SyncClock
from cogsense.fusion import AOI, FusionHub, GazeSample, RulaSample

log = logging.getLogger(__name__)

try:
    import pylsl
except Exception:  # pragma: no cover
    pylsl = None


LSL_CHANNELS = (
    "confidence", "yaw_deg", "pitch_deg", "roll_deg",
    "au04", "au07", "au01", "au02", "au14", "au45",
    "mes", "cfi", "surprise", "speech",
)


def payload_to_lsl_sample(p: dict) -> list[float]:
    nan = float("nan")
    hp = p.get("head_pose") or {}
    au = p.get("action_units") or {}
    cm = p.get("cognitive_metrics") or {}
    def g(d, k):
        v = d.get(k)
        return nan if v is None else float(v)
    return [
        float(p.get("confidence", 0.0)), g(hp, "yaw_deg"), g(hp, "pitch_deg"), g(hp, "roll_deg"),
        g(au, "au04_brow_lowerer"), g(au, "au07_lid_tightener"), g(au, "au01_inner_brow_raiser"),
        g(au, "au02_outer_brow_raiser"), g(au, "au14_dimpler"), g(au, "au45_blink_state"),
        g(cm, "mental_effort_score"), g(cm, "cognitive_friction_index"),
        g(cm, "automation_surprise_flag"), g(cm, "speech_interference_detected"),
    ]


class InboundRouter:
    """Applies inbound context messages to the fusion hub."""

    def __init__(self, fusion: FusionHub, clock: SyncClock, on_control: Callable[[str, dict], None] | None = None):
        self.fusion = fusion
        self.clock = clock
        self.on_control = on_control

    def _t(self, msg: dict) -> float:
        if "t_utc_ms" in msg:
            return self.clock.from_utc_ms(float(msg["t_utc_ms"]))
        if "t_lsl" in msg:
            return self.clock.from_lsl(float(msg["t_lsl"]))
        return self.clock.now()

    def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "gaze":
            self.fusion.add_gaze(GazeSample(self._t(msg), float(msg["x"]), float(msg["y"]), bool(msg.get("saccade", False))))
        elif kind == "rula":
            self.fusion.add_rula(RulaSample(self._t(msg), int(msg["grand"]), msg.get("neck"), msg.get("trunk")))
        elif kind == "ui_event":
            if str(msg.get("event", "")).upper() in ("TASK_COMPLETE", "TASK_COMPLETED"):
                self.fusion.add_task_completion(self._t(msg))
        elif kind == "aoi_layout":
            self.fusion.set_aois([AOI(a["name"], float(a["x"]), float(a["y"]), float(a["w"]), float(a["h"])) for a in msg["aois"]])
        elif kind == "phase":
            self.fusion.set_phase(msg.get("name"))
        elif kind == "control" and self.on_control:
            self.on_control(str(msg.get("cmd")), msg)
        else:
            log.debug("ignored inbound message: %s", kind)


class WebSocketBroadcaster:
    """Runs a websockets server on its own asyncio loop/thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, router: InboundRouter | None = None):
        self.host, self.port, self.router = host, port, router
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="cogsense-ws", daemon=True)
        self._thread.start()
        if not self._ready.wait(5.0):
            raise RuntimeError("WebSocket server failed to start")

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        import websockets

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()

        async def handler(ws):
            self._clients.add(ws)
            try:
                async for raw in ws:
                    if self.router is None:
                        continue
                    try:
                        msg = json.loads(raw)
                        for m in msg if isinstance(msg, list) else [msg]:
                            self.router.handle(m)
                    except Exception as exc:  # malformed client input must not kill the server
                        log.warning("bad inbound message: %s", exc)
            finally:
                self._clients.discard(ws)

        async with websockets.serve(handler, self.host, self.port) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    def publish(self, payload: dict) -> None:
        if not self._loop or not self._clients:
            return
        data = json.dumps(payload, separators=(",", ":"))
        self._loop.call_soon_threadsafe(self._broadcast, data)

    def _broadcast(self, data: str) -> None:
        import websockets

        websockets.broadcast(set(self._clients), data)

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=5)


class LSLPublisher:
    """Numeric AU/metric stream plus a JSON marker stream for discrete events."""

    def __init__(self, clock: SyncClock, rate_hz: float = 60.0, source_id: str = "cogsense-hfe"):
        if pylsl is None:
            raise RuntimeError("pylsl is not available")
        self.clock = clock
        info = pylsl.StreamInfo("CogSense_FACS", "FACS", len(LSL_CHANNELS), rate_hz, "float32", source_id)
        chans = info.desc().append_child("channels")
        for c in LSL_CHANNELS:
            chans.append_child("channel").append_child_value("label", c)
        self.outlet = pylsl.StreamOutlet(info)
        minfo = pylsl.StreamInfo("CogSense_Events", "Markers", 1, 0, "string", source_id + "-events")
        self.markers = pylsl.StreamOutlet(minfo)

    def publish(self, payload: dict, t: float) -> None:
        ts = self.clock.to_lsl(t)
        self.outlet.push_sample(payload_to_lsl_sample(payload), ts)
        for ev in payload.get("events", []):
            self.markers.push_sample([json.dumps(ev)], ts)


class LSLInletReader:
    """Optional: pull gaze / RULA context from existing LSL streams.

    Gaze channels are expected as [x, y, (saccade)], RULA as [grand, neck, trunk].
    """

    def __init__(self, router: InboundRouter, gaze_name: str | None = None, rula_name: str | None = None):
        if pylsl is None:
            raise RuntimeError("pylsl is not available")
        self.router = router
        self.specs = [(n, k) for n, k in ((gaze_name, "gaze"), (rula_name, "rula")) if n]
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for name, kind in self.specs:
            th = threading.Thread(target=self._pull, args=(name, kind), daemon=True, name=f"lsl-{kind}")
            th.start()
            self._threads.append(th)

    def _pull(self, name: str, kind: str) -> None:
        streams = pylsl.resolve_byprop("name", name, timeout=10.0)
        if not streams:
            log.error("LSL stream %r not found", name)
            return
        inlet = pylsl.StreamInlet(streams[0])
        while not self._stop.is_set():
            sample, ts = inlet.pull_sample(timeout=0.5)
            if sample is None:
                continue
            ts += inlet.time_correction()
            if kind == "gaze":
                msg = {"type": "gaze", "x": sample[0], "y": sample[1],
                       "saccade": bool(sample[2]) if len(sample) > 2 else False, "t_lsl": ts}
            else:
                msg = {"type": "rula", "grand": int(sample[0]),
                       "neck": int(sample[1]) if len(sample) > 1 else None,
                       "trunk": int(sample[2]) if len(sample) > 2 else None, "t_lsl": ts}
            self.router.handle(msg)

    def stop(self) -> None:
        self._stop.set()


class JSONLRecorder:
    """Session telemetry log — numeric/anonymized data only, never imagery."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def publish(self, payload: dict) -> None:
        line = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()
