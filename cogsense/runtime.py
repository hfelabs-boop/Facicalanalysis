"""Live runtime: camera → contrast normalization → MediaPipe Face Landmarker →
CogSense engine → sync bus.

Privacy (Zero Raw Video Retention): frames live only in volatile memory and
are dropped after landmarking. Video is written to disk *only* when the
runtime is built with ``audit_video=True`` AND a recording is active.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from cogsense.bus import InboundRouter, JSONLRecorder, LSLPublisher, WebSocketBroadcaster
from cogsense.calibration import Baseline
from cogsense.clock import SyncClock
from cogsense.config import CogSenseConfig
from cogsense.engine import CogSenseEngine, FaceObservation
from cogsense.fusion import FusionHub

log = logging.getLogger(__name__)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_MODEL = Path(os.environ.get("COGSENSE_MODEL", Path.home() / ".cache" / "cogsense" / "face_landmarker.task"))


def fetch_model(dest: Path = DEFAULT_MODEL) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading face landmarker model → %s", dest)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(dest)
    return dest


class ContrastNormalizer:
    """CLAHE on the LAB lightness channel (FR-2.3, 150–1500 lx tolerance)."""

    def __init__(self, clip_limit: float = 2.0, tiles: int = 8, max_width: int | None = 960):
        import cv2

        self.cv2 = cv2
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
        self.max_width = max_width

    def __call__(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (RGB frame for the landmarker, mean luminance). The frame may be
        downscaled to ``max_width``; landmarks are resolution-normalized, so this
        trades negligible precision for a large latency saving at 1080p."""
        cv2 = self.cv2
        if self.max_width and bgr.shape[1] > self.max_width:
            scale = self.max_width / bgr.shape[1]
            bgr = cv2.resize(bgr, (self.max_width, int(round(bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]
        lab[:, :, 0] = self.clahe.apply(l_ch)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return rgb, float(l_ch.mean())


class MediaPipeTracker:
    """Wrapper around the MediaPipe Tasks Face Landmarker (478 points + blendshapes)."""

    def __init__(self, model_path: str | Path, use_gpu: bool = False):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        delegate = BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def __call__(self, rgb: np.ndarray, t_s: float) -> tuple[np.ndarray | None, dict | None]:
        ts_ms = max(int(t_s * 1000), self._last_ms + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ms = ts_ms
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(image, ts_ms)
        if not res.face_landmarks:
            return None, None
        lm = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float64)
        bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0]} if res.face_blendshapes else None
        return lm, bs

    def close(self) -> None:
        self.landmarker.close()


class CogSenseRuntime:
    def __init__(self, cfg: CogSenseConfig | None = None, source: int | str = 0, model_path: str | Path | None = None,
                 ws_host: str = "127.0.0.1", ws_port: int = 8765, lsl: bool = False, record_dir: str | Path | None = None,
                 audit_video: bool = False, baseline: Baseline | None = None, use_gpu: bool = False,
                 hud_mesh_every: int = 4, width: int = 1920, height: int = 1080, fps: float = 60.0,
                 normalize_contrast: bool = True):
        self.cfg = cfg or CogSenseConfig()
        self.clock = SyncClock()
        self.fusion = FusionHub(self.cfg)
        self.engine = CogSenseEngine(self.cfg, self.fusion, baseline)
        self.source, self.width, self.height, self.fps = source, width, height, fps
        self.model_path = Path(model_path) if model_path else fetch_model()
        self.use_gpu = use_gpu
        self.router = InboundRouter(self.fusion, self.clock, self._on_control)
        self.ws = WebSocketBroadcaster(ws_host, ws_port, self.router) if ws_port is not None else None
        self.lsl = LSLPublisher(self.clock, fps) if lsl else None
        self.record_dir = Path(record_dir) if record_dir else Path("sessions")
        self.audit_video = audit_video
        self.hud_mesh_every = hud_mesh_every
        self.normalizer = ContrastNormalizer() if normalize_contrast else None
        self._recorder: JSONLRecorder | None = None
        self._video_writer = None
        self._rec_lock = threading.Lock()
        self._stop = threading.Event()
        self.recording_started_at: float | None = None

    # ------------------------------------------------------------------ control
    def _on_control(self, cmd: str, msg: dict) -> None:
        if cmd == "calibrate":
            self.engine.start_calibration(self.clock.now())
        elif cmd == "record_start":
            self.start_recording(msg.get("label"))
        elif cmd == "record_stop":
            self.stop_recording()

    def start_recording(self, label: str | None = None) -> Path:
        with self._rec_lock:
            if self._recorder:
                return self._recorder.path
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"{stamp}-{label}" if label else stamp
            self._recorder = JSONLRecorder(self.record_dir / f"{name}.jsonl")
            self.recording_started_at = self.clock.now()
            if self.audit_video:
                import cv2

                path = self.record_dir / f"{name}.audit.mp4"
                self._video_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                                     (self.width, self.height))
                log.warning("AUDIT/RECORD MODE: raw video is being written to %s", path)
            return self._recorder.path

    def stop_recording(self) -> None:
        with self._rec_lock:
            if self._recorder:
                self._recorder.close()
                self._recorder = None
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            self.recording_started_at = None

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source {self.source!r}")
        tracker = MediaPipeTracker(self.model_path, self.use_gpu)
        if self.ws:
            self.ws.start()
            log.info("WebSocket bus on ws://%s:%d", self.ws.host, self.ws.port)
        frame_id = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                t = self.clock.now()  # stamp at capture, before any processing
                if not ok:
                    break
                h, w = frame.shape[:2]
                with self._rec_lock:
                    if self._video_writer is not None:
                        self._video_writer.write(frame)
                if self.normalizer:
                    rgb, luma = self.normalizer(frame)
                else:
                    rgb, luma = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None
                del frame  # nothing retains the raw image beyond this point
                lm, bs = tracker(rgb, t)
                obs = FaceObservation(t, lm, w, h, frame_id, self.clock.to_utc_ms(t), bs, luma)
                payload = self.engine.process(obs)
                payload["clock_drift_ms"] = round(self.clock.drift_ms(), 3)
                self._publish(payload, t, lm, frame_id)
                frame_id += 1
        finally:
            cap.release()
            tracker.close()
            self.stop_recording()
            if self.ws:
                self.ws.stop()

    def _publish(self, payload: dict, t: float, lm: np.ndarray | None, frame_id: int) -> None:
        with self._rec_lock:
            if self._recorder:
                self._recorder.publish(payload)
        if self.lsl:
            self.lsl.publish(payload, t)
        if self.ws:
            msg = dict(payload)
            msg["recording"] = self.recording_started_at is not None
            if lm is not None and self.hud_mesh_every and frame_id % self.hud_mesh_every == 0:
                msg["hud_mesh"] = np.round(lm[:, :2], 4).tolist()  # anonymized 2D vectors, HUD only
            self.ws.publish(msg)

    def stop(self) -> None:
        self._stop.set()
