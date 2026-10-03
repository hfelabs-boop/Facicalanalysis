# CogSense HFE Module

**Facial Action Coding & Micro-Expression Cognitive Engine.** This is a standalone micro-service and plug-in for the Ergonomic Pose (RULA) & Eye-Tracking Evaluation Suite.

CogSense reads the shared RGB camera stream and tracks a 478-point face mesh. From it, the module extracts upper-face Action Units (**AU4, AU7, AU1+2, AU14, AU43/45**) and computes two cognitive measures:

- a continuous **Mental Effort Score (MES)**
- a **Cognitive Friction Index (CFI)**

It then fuses these with gaze AOIs and RULA posture scores. Output is a timestamped telemetry stream sent over WebSocket and LSL. The module also includes a live analytics console and an automated PDF/CSV evaluation report.

```
camera ─► CLAHE contrast norm ─► MediaPipe Face Landmarker (478 pts + blendshapes)
              │
              ▼
   head-pose & rigidity normalization ──► speech/mastication mask
              │                                   │
              ▼                                   ▼
        AU regression core (AU1, AU2, AU4, AU7, AU14, AU43/45)
              │
              ▼
   MES · CFI · automation surprise ◄── fusion hub ◄── gaze/AOI · RULA · C2 UI events
              │
              ▼
   sync bus: WebSocket JSON · LSL (FACS + Markers) · JSONL session log
```

## Quick start

```bash
pip install -e ".[all]"            # numpy, websockets, mediapipe, opencv, pylsl, reportlab
cogsense fetch-model               # downloads face_landmarker.task to ~/.cache/cogsense

# Live camera with dashboard on http://127.0.0.1:8080 and WebSocket bus on :8765
cogsense run --source 0 --lsl

# No camera? Stream a scripted C2 scenario through the real engine
cogsense simulate --realtime --speed 2 --loop     # then open http://127.0.0.1:8080

# Offline session → evaluation dossier
cogsense simulate --out sessions/demo.jsonl
cogsense report sessions/demo.jsonl --pdf reports/demo.pdf --csv reports/demo/
```

On a headless Linux host, MediaPipe needs the EGL/GLES runtime: `apt-get install libegl1 libgles2`.

## iPhone / phone camera

Open the deployed console (or any https host serving `dashboard/`) in **Safari on the iPhone**. On phones it opens in **Camera** mode.

1. Tap **Start camera** and allow camera access. The front camera is used.
2. Wait for the face tracker to load. The first run downloads about 14 MB, then it is cached.
3. Tap **Calibrate (15 s)** and hold a relaxed, neutral face. The status reads `CALIBRATED ✓` when finished.
4. The HUD shows the mesh, AU bars, MES and CFI live. Use **Export buffer (.jsonl)** at the end, then run `cogsense report` for the dossier.

How it works: `dashboard/camera.js` runs the MediaPipe Face Landmarker (WASM) in the browser. `dashboard/engine.js` is a line-for-line port of the Python engine, checked frame by frame against it by `tests/test_js_parity.py`. **Video never leaves the phone.** Nothing is uploaded or stored, and only derived numbers are held in memory until you export.

iPhone notes:

- **iOS 16.4 or newer, in Safari.** Camera access needs https. Plain `http://<laptop-ip>` pages are blocked by iOS, which is why this runs on-device instead of streaming to the Python service.
- **Open the link in Safari itself.** In-app browsers (Instagram, Facebook, etc.) block the camera, and the page says so.
- **Permission denied?** Tap *aA* in the address bar → Website Settings → Camera → Allow, then reload.
- **Keep the page in the foreground.** iOS stops the camera when Safari is backgrounded, and the page asks you to tap Start camera again. The screen is kept awake while running.
- **Phone mounted at arm's length, face centred.** Keep the face inside ±30° yaw and ±20° pitch, as for any camera.
- **Self-host the runtime** for closed networks: serve MediaPipe's `vision_bundle.mjs` plus `wasm/` and the model yourself, then open the page with `?mp=<base-url>&model=<url>`.
- **No eye tracker or RULA on a phone.** Gaze/AOI/posture fusion needs the workstation service. The phone gives you AUs, MES, CFI, blink and surprise events.
- **Mac or Windows with the iPhone as a webcam (Continuity Camera and similar):** the iPhone appears as a normal camera, so `cogsense run --source <index>` works directly with the full engine, including the LSL and WebSocket bus.

## Operator workflow

1. Start `cogsense run`. The engine builds a **provisional** baseline from the first 30 confident frames, so output begins immediately.
2. Click **Calibrate** in the console, or send `{"type":"control","cmd":"calibrate"}`. The operator holds a relaxed, neutral face for **15 s** (FR-1.3). Blinks, speech and low-confidence frames are excluded. The engine stores median resting features, blendshape rest values and the resting blink rate. Status then becomes `CALIBRATED`.
3. Have the eye tracker, the RULA service and the C2 UI push context over the bus (see below), and send the screen's AOI layout.
4. Click **Record**, or pass `--record`, to write the anonymized session log. Open `cogsense report` afterwards for the dossier, or load the JSONL in the console's **Playback** mode. Playback can sync an operator video and a screen recording to the timeline scrubber.

## Requirements traceability

| Req | Implementation |
|---|---|
| FR-1.1 AU scope | `au_engine.py`, `blink.py`. Geometric evidence from pose-normalized landmarks, optionally fused with MediaPipe blendshapes or a trained `LinearAUModel`. AU45 = blink state; AU43 = closure ≥ 500 ms (microsleep). |
| FR-1.2 0–1 floats, ≥30 Hz | All AU outputs are clipped to [0, 1]. Engine cost is ~0.3 ms/frame, so the camera rate decides throughput. |
| FR-1.3 15 s baseline | `calibration.py` (`BaselineCalibrator`, `Baseline.save/load`, `--baseline`). |
| FR-2.1 pose invariance | `geometry.py` builds a face frame from rigid landmarks, rotates the mesh into it and scales by IOD. Tests cover yaw ±30°, pitch ±20°, roll ±20°. Confidence decays outside that range. |
| FR-2.2 speech/chewing | `speech.py` watches lip-aperture and jaw variance. When active it sets `speech_interference_detected`, attenuates AU14 ×0.3 and multiplies confidence by 0.85. |
| FR-2.3 illumination | `runtime.ContrastNormalizer` applies CLAHE on LAB lightness. Frames with extreme luminance get reduced confidence. |
| FR-3.1 MES | `metrics.py` (formula below). |
| FR-3.2 CFI | `metrics.py`: a continuous index plus discrete `CFI_EVENT`s. |
| FR-3.3 automation surprise | `metrics.py`: `AUTOMATION_SURPRISE` event and a 1 s latched flag. |
| FR-4.1 sync bus | `clock.py` stamps every frame at capture on one monotonic clock and slews offsets to UTC and LSL. `bus.py` handles WebSocket, LSL (`CogSense_FACS` 16-ch + `CogSense_Events` markers) and JSONL. |
| FR-4.2 gaze/AOI | `fusion.FusionHub`: nearest-in-time gaze (staleness 250 ms), AOI hit-test and saccade rate. |
| FR-4.3 biomechanical correlator | `fusion.correlate`: edge-triggered `COMPOUND_POSTURE_RISK` when RULA neck ≥ 3 or trunk ≥ 3 co-occurs with AU7 ≥ 0.40 or AU4 ≥ 0.50. |
| NFR privacy | Frames stay in memory only. Video is written **only** with `--audit-video` while recording. Session logs hold numbers only; HUD mesh points go over WebSocket and are never logged. |
| NFR graceful fallback | No face, extreme pose, implausible geometry, one eye hidden or a tracker glitch → `tracking_status: LOST`, `confidence < 0.3`, and `action_units` / `cognitive_metrics` set to `null`. Stale values are never sent. |
| §7 console | `dashboard/`: live HUD (mesh/AU bars/MES/CFI), synchronized MES+RULA timeline with phase bands and event markers, gaze heatmap and scan path, findings feed, playback with video sync. |
| §7 report | `report.py`: mean/peak MES per mission phase, top-5 AOIs by cumulative CFI, posture × AU7 cross-tab, event log, findings → CSV + PDF. |

## Derived metrics

The full step-by-step reference (every AU, MES, CFI, events, confidence, report figures, with worked examples) is the console's **Docs** page: `dashboard/docs.html`, served at `/docs`. It includes a **Research evidence** section that sets each formula against published work and lists where the evidence disagrees with the defaults. The summary below is the short form.

**Mental Effort Score (0–100).** The PRD left the formula blank, so this is the defined implementation:

```
e_t   = 0.7·AU4_t + 0.3·S_t,      S_t = clip(1 − blink_rate_t / resting_blink_rate, 0, 1)
MES_t = 100 · mean(e) over a rolling 5 s window
```

Blink rate comes from a rolling 30 s window. If it isn't available yet, `e_t = AU4_t`.

**Cognitive Friction Index (0–1)**

```
CFI_t = clip(AU4_t·(0.6 + 0.4·C_t) + 0.15·R_t, 0, 1)    (halved for 2 s after TASK_COMPLETE)
C_t   = max(clip(AU14_t/0.30), clip(saccade_rate/3 Hz))     co-occurrence
R_t   = clip(ΔAU4 over 200 ms / 0.25)                       spike rise
```

A `CFI_EVENT` fires when **all** of the following hold:

- AU4 exceeds 0.65 after rising ≥ 0.25 within 200 ms
- AU14 ≥ 0.30, or saccade rate ≥ 3 Hz
- no task completion in the previous 2 s

Events have a 1 s refractory period. The console and report mark CFI > 0.70.

**Automation surprise.** Fires when mean(AU1, AU2) > 0.50 is followed within 800 ms by an AU4 surge (≥ 0.50, rising ≥ 0.20 within 200 ms).

All thresholds live in `CogSenseConfig`. Override them with `cogsense run --config overrides.json`.

## Telemetry payload

The payload is a superset of PRD §6. These fields are additive: `head_pose.within_operating_range`, `action_units.au43_eyes_closed`, `blink`, `fatigue` (PERCLOS and blink dynamics), `fusion_context.{gaze_px,saccade_rate_hz,rula_trunk_score,mission_phase,pupil}`, `calibration`, `events`, `processing_ms` and `clock_drift_ms`.

```json
{
  "timestamp_utc_ms": 1790947625120, "frame_id": 48102,
  "tracking_status": "LOCKED", "confidence": 0.94,
  "head_pose": {"yaw_deg": 4.2, "pitch_deg": -6.1, "roll_deg": 0.8, "within_operating_range": true},
  "action_units": {"au04_brow_lowerer": 0.72, "au07_lid_tightener": 0.45, "au01_inner_brow_raiser": 0.05,
                   "au02_outer_brow_raiser": 0.02, "au14_dimpler": 0.38, "au45_blink_state": 0, "au43_eyes_closed": false},
  "blink": {"rate_per_min": 11.0, "last_duration_ms": 160.0, "closure_ms": 0.0},
  "cognitive_metrics": {"mental_effort_score": 68.4, "cognitive_friction_index": 0.81,
                        "automation_surprise_flag": false, "speech_interference_detected": false},
  "fusion_context": {"active_aoi": "TACTICAL_RADAR_WIDGET_PRIMARY", "gaze_px": [612.0, 388.5], "saccade_rate_hz": 4,
                     "pupil": {"diameter_mm": 3.81, "baseline_mm": 3.50, "change_mm": 0.31, "change_pct": 8.86, "reliable": true},
                     "rula_grand_score": 5, "rula_neck_score": 3, "rula_trunk_score": 2,
                     "mission_phase": "SWARM_ENGAGEMENT", "correlated_insight": "HIGH_COGNITIVE_STRAIN_ON_COMPLEX_WIDGET"},
  "fatigue": {"perclos": 0.07, "window_s": 60.0, "coverage": 0.99, "blink_count": 16, "mean_blink_ms": 148.0, "long_closures": 0},
  "calibration": {"status": "CALIBRATED", "progress": 1.0},
  "events": [{"type": "CFI_EVENT", "cfi": 0.81, "au04": 0.72, "trigger": "AU14", "aoi": "TACTICAL_RADAR_WIDGET_PRIMARY"}]
}
```

`correlated_insight` takes one of these values:

- `FATIGUE_MICROSLEEP_RISK`
- `AUTOMATION_SURPRISE`
- `POSTURE_DRIVEN_VISUAL_COMPENSATION`
- `HIGH_COGNITIVE_STRAIN_ON_COMPLEX_WIDGET`
- `HIGH_COGNITIVE_FRICTION`
- `VISUAL_STRAIN_DISPLAY_LEGIBILITY`
- `POSTURAL_LOAD_WITH_COGNITIVE_STRAIN`
- `SUSTAINED_MENTAL_EFFORT`
- `null`

## Feeding context into the bus

Send JSON to `ws://host:8765`, either single objects or arrays. `t_utc_ms` (or `t_lsl`) is optional; messages without it are stamped on arrival.

```json
{"type": "gaze", "x": 812, "y": 440, "saccade": false, "t_utc_ms": 1790947625101}
{"type": "rula", "grand": 5, "neck": 3, "trunk": 2}
{"type": "pupil", "diameter_mm": 3.42, "valid": true, "luminance": 0.5}
{"type": "ui_event", "event": "TASK_COMPLETE"}
{"type": "aoi_layout", "aois": [{"name": "RADAR", "x": 0, "y": 0, "w": 800, "h": 600}]}
{"type": "phase", "name": "SWARM_ENGAGEMENT"}
{"type": "control", "cmd": "calibrate" | "record_start" | "record_stop"}
```

Gaze, RULA and pupil can also be pulled from existing LSL streams via `bus.LSLInletReader`. The pupil message takes diameter in **millimetres** from your infrared eye tracker; a plain RGB camera cannot measure it. A sample client is in `examples/push_context.py`.

## Phase 2: training and validating against FACS benchmarks

```bash
cogsense extract-features SN001.avi --subject SN001 -o feats/SN001.csv
# join DISFA/BP4D per-frame labels as columns au01, au02, au04, au07, au14 (0–5 intensity)
cogsense validate feats/*.csv --save-model au_model.json        # leave-one-subject-out F1, target ≥ 0.82
```

The trained logistic weights load through `LinearAUModel` and replace the hand-tuned geometric mapping. DISFA and BP4D must be obtained under their own licences; they are not included.

## Development

```bash
pip install -e ".[dev]"
pytest -q
cogsense bench --mediapipe     # per-frame latency: engine, CLAHE, landmarker
```

## Status against the acceptance criteria

- **Implemented and covered by tests (108 tests):**
  - AU extraction and pose invariance on synthetic meshes
  - calibration, speech mask, graceful fallback
  - MES / CFI / surprise logic and fusion/compound risk
  - WebSocket round-trip, LSL outlet, clock slewing
  - report and PDF export
  - metric reference page (`dashboard/docs.html`) kept in step with the config constants
  - browser/Node engine parity with the Python engine, frame by frame
  - zero-video-retention behaviour
- **Measured in a 4-core CPU container, with no GPU and no real face:**
  - CogSense engine: ~0.3 ms/frame
  - CLAHE + downscale of a 1080p frame: ~5 ms
  - Latency of the landmarker *on a real face* still needs measuring on the target workstation (`cogsense bench --mediapipe`, plus a live run).
- **Not yet done:**
  - **Phase 2.** Training on DISFA/BP4D and reaching the F1 ≥ 0.82 target. The tooling is in place, but the datasets are licensed and not available here. The default AU mapping is a hand-tuned geometric model plus MediaPipe blendshapes. It is not yet validated against FACS-coded ground truth.
  - **Phase 3.** The 60-minute drift trial against live RULA and gaze hardware. Drift is monitored per frame (`clock_drift_ms`).
  - **Phase 4.** Practitioner usability evaluation.
- **Known limitations:**
  - AU14 is a weak geometric signal from a monocular mesh; it relies more on blendshapes and the trained model.
  - Occlusion detection is heuristic, because MediaPipe extrapolates hidden landmarks.
  - Head-pose ranges beyond ±30° yaw degrade by design.
