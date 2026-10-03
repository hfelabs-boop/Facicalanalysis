"""Command-line entry point: ``cogsense <command>``."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import logging
import sys
import threading
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


def _serve_dashboard(port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DASHBOARD_DIR))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    logging.info("dashboard at http://127.0.0.1:%d/", port)


def cmd_run(a) -> None:
    from cogsense.calibration import Baseline
    from cogsense.config import CogSenseConfig
    from cogsense.runtime import CogSenseRuntime

    cfg = CogSenseConfig.load(a.config) if a.config else CogSenseConfig()
    baseline = Baseline.load(a.baseline) if a.baseline else None
    source = int(a.source) if str(a.source).isdigit() else a.source
    rt = CogSenseRuntime(cfg, source, a.model, a.host, a.port, a.lsl, a.record_dir, a.audit_video, baseline,
                         a.gpu, width=a.width, height=a.height, fps=a.fps)
    if a.dashboard_port:
        _serve_dashboard(a.dashboard_port)
    if a.record:
        rt.start_recording(a.label)
    try:
        rt.run()
    except KeyboardInterrupt:
        rt.stop()


def cmd_simulate(a) -> None:
    from cogsense.bus import JSONLRecorder, WebSocketBroadcaster
    from cogsense.simulate import ScenarioSimulator, run_realtime

    sim = ScenarioSimulator(fps=a.fps, seed=a.seed)
    if a.out and not a.realtime:
        rec = JSONLRecorder(a.out)
        for p in sim.run():
            rec.publish(p)
        rec.close()
        print(f"wrote {a.out} ({sim.duration_s:.0f} s scenario)")
        return
    sinks = []
    ws = WebSocketBroadcaster(a.host, a.port)
    ws.start()
    sinks.append(ws)
    if a.out:
        sinks.append(JSONLRecorder(a.out))
    if a.dashboard_port:
        _serve_dashboard(a.dashboard_port)
    print(f"streaming simulated telemetry on ws://{a.host}:{ws.port}")
    try:
        while True:
            run_realtime(sim, sinks, a.speed)
            if not a.loop:
                break
            sim = ScenarioSimulator(fps=a.fps, seed=a.seed + 1)
    except KeyboardInterrupt:
        pass
    finally:
        ws.stop()


def cmd_report(a) -> None:
    from cogsense.report import analyze, export_csv, export_pdf, load_session

    rep = analyze(load_session(a.session))
    if a.csv:
        for p in export_csv(rep, a.csv):
            print("wrote", p)
    if a.pdf:
        print("wrote", export_pdf(rep, a.pdf))
    if not a.csv and not a.pdf:
        print(json.dumps({"phases": rep.phases, "top_aois": rep.top_aois, "crosstab": rep.crosstab,
                          "findings": rep.findings}, indent=2))


def cmd_fetch_model(a) -> None:
    from cogsense.runtime import DEFAULT_MODEL, fetch_model

    print(fetch_model(Path(a.dest) if a.dest else DEFAULT_MODEL))


def cmd_bench(a) -> None:
    from cogsense.bench import run_bench

    print(json.dumps(run_bench(a.frames, a.model, a.mediapipe), indent=2))


def cmd_extract(a) -> None:
    from cogsense.runtime import fetch_model
    from cogsense.validation import extract_video_features

    n = extract_video_features(a.video, a.model or fetch_model(), a.subject, a.out)
    print(f"wrote {n} frames to {a.out}")


def cmd_validate(a) -> None:
    from cogsense.validation import cross_validate, fit_all, load_labeled, save_model

    X, Y, subj = load_labeled(a.csv, a.threshold)
    res = cross_validate(X, Y, subj)
    print(json.dumps({"leave_one_subject_out_f1": res, "target": 0.82,
                      "passed": all(v >= 0.82 for v in res.values())}, indent=2))
    if a.save_model:
        save_model(fit_all(X, Y), a.save_model)
        print("saved", a.save_model)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cogsense", description="CogSense HFE facial action & cognitive engine")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="live camera processing")
    r.add_argument("--source", default="0", help="camera index or video file")
    r.add_argument("--model", help="face_landmarker.task (downloaded if omitted)")
    r.add_argument("--config", help="JSON config overrides")
    r.add_argument("--baseline", help="saved neutral baseline JSON")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--lsl", action="store_true", help="publish LSL streams")
    r.add_argument("--record", action="store_true", help="start telemetry recording immediately")
    r.add_argument("--label", help="recording label")
    r.add_argument("--record-dir", default="sessions")
    r.add_argument("--audit-video", action="store_true",
                   help="AUDITING/RECORD MODE: also write raw video while recording (off by default)")
    r.add_argument("--gpu", action="store_true")
    r.add_argument("--width", type=int, default=1920)
    r.add_argument("--height", type=int, default=1080)
    r.add_argument("--fps", type=float, default=60.0)
    r.add_argument("--dashboard-port", type=int, default=8080)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("simulate", help="scripted scenario through the real engine")
    s.add_argument("--out", help="write telemetry JSONL")
    s.add_argument("--realtime", action="store_true", help="stream at wall-clock pace over WebSocket")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--loop", action="store_true")
    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--dashboard-port", type=int, default=8080)
    s.set_defaults(fn=cmd_simulate)

    rp = sub.add_parser("report", help="HFE evaluation report from a session JSONL")
    rp.add_argument("session")
    rp.add_argument("--csv", help="output directory for CSV dossier")
    rp.add_argument("--pdf", help="output PDF path")
    rp.set_defaults(fn=cmd_report)

    f = sub.add_parser("fetch-model", help="download the MediaPipe face landmarker model")
    f.add_argument("--dest")
    f.set_defaults(fn=cmd_fetch_model)

    b = sub.add_parser("bench", help="measure per-frame latency")
    b.add_argument("--frames", type=int, default=600)
    b.add_argument("--mediapipe", action="store_true", help="include landmark inference on a 1080p frame")
    b.add_argument("--model")
    b.set_defaults(fn=cmd_bench)

    e = sub.add_parser("extract-features", help="video → per-frame feature CSV (for model training)")
    e.add_argument("video")
    e.add_argument("--subject", required=True)
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--model")
    e.set_defaults(fn=cmd_extract)

    v = sub.add_parser("validate", help="leave-one-subject-out AU F1 on labelled feature CSVs")
    v.add_argument("csv", nargs="+")
    v.add_argument("--threshold", type=float, default=2.0, help="intensity at/above which an AU counts as present")
    v.add_argument("--save-model")
    v.set_defaults(fn=cmd_validate)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
