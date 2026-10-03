/* CogSense engine — browser/Node port of the Python core (cogsense/*.py).
 *
 * Same maths, same defaults, same payload schema, so a phone can run the whole
 * pipeline locally (camera → MediaPipe → this engine) and no video ever leaves
 * the device. tests/test_js_parity.py checks it frame-by-frame against Python.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.CogSense = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ------------------------------------------------------------------ config
  const DEFAULTS = {
    target_hz: 60.0, calibration_seconds: 15.0, calibration_min_frames: 90,
    scales: { brow_height: 0.08, brow_gap: 0.10, outer_brow_height: 0.09, eye_aperture_ratio: 0.42,
      lower_lid_raise: 0.05, lip_corner_depth: 0.06, mouth_width: 0.06 },
    blendshape_weight: 0.5, smoothing_tau_s: 0.04,
    blink_close_ratio: 0.45, blink_open_ratio: 0.60, blink_rate_window_s: 30.0, microsleep_ms: 500.0,
    default_blink_rate_per_min: 17.0,
    speech_window_s: 0.6, speech_aperture_std: 0.018, speech_open_delta: 0.07,
    speech_lower_face_attenuation: 0.3, speech_confidence_factor: 0.85,
    yaw_limit_deg: 30.0, pitch_limit_deg: 20.0, roll_limit_deg: 20.0, pose_falloff_deg: 20.0,
    lost_confidence: 0.3, locked_confidence: 0.6, min_iod_px: 25.0,
    mes_window_s: 5.0, mes_au4_weight: 0.7,
    cfi_au4_threshold: 0.65, cfi_rise_window_s: 0.2, cfi_rise_min: 0.25, cfi_au14_threshold: 0.30,
    cfi_saccade_rate_hz: 3.0, cfi_task_completion_window_s: 2.0, cfi_event_refractory_s: 1.0,
    cfi_marker_threshold: 0.70,
    surprise_brow_threshold: 0.50, surprise_window_s: 0.8, surprise_au4_surge: 0.50, surprise_latch_s: 1.0,
    perclos_window_s: 60.0, perclos_closed_openness: 0.20, perclos_min_coverage_s: 10.0, perclos_alert: null, fatigue_max_dt_s: 0.1,
    rula_neck_threshold: 3, rula_trunk_threshold: 3, compound_au7_threshold: 0.40, compound_au4_threshold: 0.50,
  };

  function makeConfig(overrides) {
    const cfg = Object.assign({}, DEFAULTS, overrides || {});
    cfg.scales = Object.assign({}, DEFAULTS.scales, (overrides && overrides.scales) || {});
    return cfg;
  }

  // ------------------------------------------------------------------ landmarks
  const LM = {
    R_EYE_OUTER: 33, R_EYE_INNER: 133, L_EYE_OUTER: 263, L_EYE_INNER: 362,
    R_EYE_LOWER: [144, 145, 153], L_EYE_LOWER: [373, 374, 380],
    R_EYE_EAR: [33, 160, 158, 133, 153, 144], L_EYE_EAR: [362, 385, 387, 263, 373, 380],
    R_BROW_INNER: [107, 55], L_BROW_INNER: [336, 285], R_BROW_OUTER: [70, 46], L_BROW_OUTER: [300, 276],
    NOSE_BRIDGE: 168, NOSE_TIP: 1, FOREHEAD: 10, CHIN: 152, FACE_R: 234, FACE_L: 454,
    MOUTH_R: 61, MOUTH_L: 291, LIP_UPPER_OUTER: 0, LIP_LOWER_OUTER: 17, LIP_UPPER_INNER: 13, LIP_LOWER_INNER: 14,
    MIN_LANDMARKS: 468,
  };
  const FEATURE_NAMES = [
    "brow_inner_h_r", "brow_inner_h_l", "brow_outer_h_r", "brow_outer_h_l", "brow_gap",
    "ear_r", "ear_l", "lower_lid_r", "lower_lid_l", "lip_corner_depth", "mouth_width",
    "mouth_aperture", "jaw_drop",
  ];
  const BLENDSHAPE_KEYS = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "mouthDimpleLeft", "mouthDimpleRight",
  ];
  const AU_KEYS = ["au01", "au02", "au04", "au07", "au14"];

  // ------------------------------------------------------------------ helpers
  const clip01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
  const rnd = (x, n) => { const f = Math.pow(10, n); return Math.round(x * f) / f; };
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => Math.sqrt(dot(a, a));
  const unit = (a) => { const n = norm(a); return n > 1e-9 ? [a[0] / n, a[1] / n, a[2] / n] : a; };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const meanPts = (p, idx) => {
    const s = [0, 0, 0];
    for (const i of idx) { s[0] += p[i][0]; s[1] += p[i][1]; s[2] += p[i][2]; }
    return [s[0] / idx.length, s[1] / idx.length, s[2] / idx.length];
  };
  const median = (arr) => {
    const a = Array.from(arr).sort((x, y) => x - y), n = a.length;
    return n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
  };
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const nowMs = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

  class EMA {
    constructor(tau) { this.tau = tau; this.value = null; this.t = null; }
    update(x, t) {
      if (this.value === null || this.t === null || this.tau <= 0) this.value = x;
      else this.value += (1 - Math.exp(-Math.max(t - this.t, 0) / this.tau)) * (x - this.value);
      this.t = t;
      return this.value;
    }
    reset() { this.value = null; this.t = null; }
  }

  class TimeWindow {
    constructor(seconds) { this.seconds = seconds; this.items = []; }
    add(t, v) { this.items.push([t, v]); this.trim(t); }
    trim(now) { let k = 0; while (k < this.items.length && now - this.items[k][0] > this.seconds) k++; if (k) this.items.splice(0, k); }
    mean() { return this.items.length ? mean(this.items.map((x) => x[1])) : null; }
    std() {
      const n = this.items.length;
      if (n < 2) return 0;
      const m = this.mean();
      return Math.sqrt(this.items.reduce((s, x) => s + (x[1] - m) ** 2, 0) / (n - 1));
    }
  }

  // ------------------------------------------------------------------ geometry (FR-2.1)
  function eulerFromRotation(R) {
    const d = 180 / Math.PI;
    return [Math.atan2(R[0][2], R[2][2]) * d, Math.asin(Math.max(-1, Math.min(1, -R[1][2]))) * d,
      Math.atan2(R[1][0], R[1][1]) * d];
  }

  /** landmarks: array of [x, y, z] in MediaPipe-normalized coords; w/h: frame size in px. */
  function normalizeFace(lm, w, h) {
    const p = lm.map((q) => [q[0] * w, q[1] * h, q[2] * w]);
    const rEye = meanPts(p, [LM.R_EYE_OUTER, LM.R_EYE_INNER]);
    const lEye = meanPts(p, [LM.L_EYE_OUTER, LM.L_EYE_INNER]);
    const iod = norm(sub(lEye, rEye));
    const a = sub(p[LM.L_EYE_OUTER], p[LM.R_EYE_OUTER]);
    const b = sub(p[LM.L_EYE_INNER], p[LM.R_EYE_INNER]);
    const c = sub(p[LM.FACE_L], p[LM.FACE_R]);
    const ex = unit([a[0] + b[0] + 0.5 * c[0], a[1] + b[1] + 0.5 * c[1], a[2] + b[2] + 0.5 * c[2]]);
    const down = sub(p[LM.CHIN], p[LM.FOREHEAD]);
    const k = dot(down, ex);
    const ey = unit([down[0] - k * ex[0], down[1] - k * ex[1], down[2] - k * ex[2]]);
    const ez = cross(ex, ey);
    const R = [[ex[0], ey[0], ez[0]], [ex[1], ey[1], ez[1]], [ex[2], ey[2], ez[2]]];
    const o = p[LM.NOSE_BRIDGE], s = Math.max(iod, 1e-6);
    const pts = p.map((q) => {
      const d = [q[0] - o[0], q[1] - o[1], q[2] - o[2]];
      return [dot(d, ex) / s, dot(d, ey) / s, dot(d, ez) / s];
    });
    const [yaw, pitch, roll] = eulerFromRotation(R);
    return { points: pts, yaw, pitch, roll, iod };
  }

  // ------------------------------------------------------------------ features
  function ear(p, idx) {
    const [p1, p2, p3, p4, p5, p6] = idx.map((i) => p[i]);
    const horiz = norm(sub(p1, p4));
    if (horiz < 1e-9) return 0;
    return (norm(sub(p2, p6)) + norm(sub(p3, p5))) / (2 * horiz);
  }

  function extractFeatures(p) {
    const rCornerY = (p[LM.R_EYE_OUTER][1] + p[LM.R_EYE_INNER][1]) / 2;
    const lCornerY = (p[LM.L_EYE_OUTER][1] + p[LM.L_EYE_INNER][1]) / 2;
    const rInner = meanPts(p, LM.R_BROW_INNER), lInner = meanPts(p, LM.L_BROW_INNER);
    const rOuter = meanPts(p, LM.R_BROW_OUTER), lOuter = meanPts(p, LM.L_BROW_OUTER);
    const lipC = [0, 1, 2].map((i) => (p[LM.LIP_UPPER_OUTER][i] + p[LM.LIP_LOWER_OUTER][i]) / 2);
    const corners = [0, 1, 2].map((i) => (p[LM.MOUTH_R][i] + p[LM.MOUTH_L][i]) / 2);
    return {
      brow_inner_h_r: rCornerY - rInner[1],
      brow_inner_h_l: lCornerY - lInner[1],
      brow_outer_h_r: p[LM.R_EYE_OUTER][1] - rOuter[1],
      brow_outer_h_l: p[LM.L_EYE_OUTER][1] - lOuter[1],
      brow_gap: Math.abs(lInner[0] - rInner[0]),
      ear_r: ear(p, LM.R_EYE_EAR),
      ear_l: ear(p, LM.L_EYE_EAR),
      lower_lid_r: meanPts(p, LM.R_EYE_LOWER)[1] - rCornerY,
      lower_lid_l: meanPts(p, LM.L_EYE_LOWER)[1] - lCornerY,
      lip_corner_depth: corners[2] - lipC[2],
      mouth_width: norm(sub(p[LM.MOUTH_L], p[LM.MOUTH_R])),
      mouth_aperture: norm(sub(p[LM.LIP_LOWER_INNER], p[LM.LIP_UPPER_INNER])),
      jaw_drop: norm(sub(p[LM.CHIN], p[LM.NOSE_TIP])),
    };
  }

  // ------------------------------------------------------------------ calibration (FR-1.3)
  class BaselineCalibrator {
    constructor(duration, minFrames) {
      this.duration = duration; this.minFrames = minFrames; this.state = "IDLE";
      this.t0 = null; this.feats = []; this.bs = []; this.blinks = 0;
    }
    start(t) { this.state = "COLLECTING"; this.t0 = t; this.feats = []; this.bs = []; this.blinks = 0; }
    progress(t) {
      if (this.state === "COMPLETE") return 1;
      if (this.state !== "COLLECTING" || this.t0 === null) return 0;
      return Math.min((t - this.t0) / this.duration, 1);
    }
    add(t, feats, bs, eyesClosed, blinkOnset, usable) {
      if (this.state !== "COLLECTING") return null;
      if (blinkOnset) this.blinks++;
      if (usable && !eyesClosed) {
        this.feats.push(FEATURE_NAMES.map((k) => feats[k]));
        if (bs) this.bs.push(bs);
      }
      if (this.progress(t) < 1 || this.feats.length < this.minFrames) return null;
      return this.finish(t);
    }
    finish(t) {
      const features = {}, mad = {};
      FEATURE_NAMES.forEach((k, j) => {
        const col = this.feats.map((r) => r[j]), m = median(col);
        features[k] = m; mad[k] = median(col.map((v) => Math.abs(v - m)));
      });
      const blendshapes = {};
      for (const k of BLENDSHAPE_KEYS) {
        const vals = this.bs.filter((b) => k in b).map((b) => b[k]);
        if (vals.length) blendshapes[k] = median(vals);
      }
      const elapsedMin = Math.max((t - (this.t0 === null ? t : this.t0)) / 60, 1e-6);
      this.state = "COMPLETE";
      return { features, feature_mad: mad, blendshapes, blink_rate_per_min: this.blinks >= 2 ? this.blinks / elapsedMin : null,
        frames: this.feats.length, is_default: false };
    }
  }

  // ------------------------------------------------------------------ blink (AU43/45)
  class BlinkDetector {
    constructor(cfg) {
      this.close = cfg.blink_close_ratio; this.open = cfg.blink_open_ratio;
      this.window = cfg.blink_rate_window_s; this.microMs = cfg.microsleep_ms;
      this.closed = false; this.closedSince = null; this.onsets = []; this.tFirst = null; this.lastBlinkMs = null;
    }
    update(t, earRatio) {
      if (this.tFirst === null) this.tFirst = t;
      let onset = false, completed = null;
      if (!this.closed && earRatio < this.close) { this.closed = true; onset = true; this.closedSince = t; this.onsets.push(t); }
      else if (this.closed && earRatio > this.open) {
        this.closed = false;
        if (this.closedSince !== null) this.lastBlinkMs = completed = (t - this.closedSince) * 1000;
        this.closedSince = null;
      }
      while (this.onsets.length && t - this.onsets[0] > this.window) this.onsets.shift();
      const closureMs = this.closed && this.closedSince !== null ? (t - this.closedSince) * 1000 : 0;
      const observed = Math.min(t - this.tFirst, this.window);
      return { closed: this.closed, onset, closure_ms: closureMs, last_blink_ms: this.lastBlinkMs,
        rate_per_min: observed >= 5 ? (this.onsets.length * 60) / observed : null, microsleep: closureMs >= this.microMs,
        completed_ms: completed };
    }
    resetGap() { this.closed = false; this.closedSince = null; }
  }

  // ------------------------------------------------------------------ fatigue (PERCLOS-style closure; separate from MES)
  class FatigueTracker {
    constructor(cfg) { this.cfg = cfg; this.segments = []; this.blinks = []; this.prevT = null; this.tFirst = null; this.lastAlert = -1e9; }
    markGap() { this.prevT = null; }
    update(t, earRatio, blink) {
      const cfg = this.cfg;
      if (this.tFirst === null) this.tFirst = t;
      if (this.prevT !== null) {
        const dt = Math.min(Math.max(t - this.prevT, 0), cfg.fatigue_max_dt_s);
        this.segments.push([t, dt, earRatio <= cfg.perclos_closed_openness]);
      }
      this.prevT = t;
      if (blink.completed_ms !== null) this.blinks.push([t, blink.completed_ms]);
      let k = 0; while (k < this.segments.length && t - this.segments[k][0] > cfg.perclos_window_s) k++; if (k) this.segments.splice(0, k);
      k = 0; while (k < this.blinks.length && t - this.blinks[k][0] > cfg.perclos_window_s) k++; if (k) this.blinks.splice(0, k);
      let valid = 0, closed = 0;
      for (const [, dt, c] of this.segments) { valid += dt; if (c) closed += dt; }
      const elapsed = Math.min(t - this.tFirst, cfg.perclos_window_s);
      const coverage = elapsed > 0 ? Math.min(valid / elapsed, 1) : 0;
      const perclos = valid >= cfg.perclos_min_coverage_s ? closed / valid : null;
      const durs = this.blinks.map((b) => b[1]);
      const out = { perclos: perclos === null ? null : rnd(perclos, 3), window_s: cfg.perclos_window_s, coverage: rnd(coverage, 3),
        blink_count: durs.length, mean_blink_ms: durs.length ? rnd(mean(durs), 1) : null,
        long_closures: durs.filter((d) => d >= cfg.microsleep_ms).length };
      const events = [];
      if (cfg.perclos_alert !== null && perclos !== null && perclos >= cfg.perclos_alert && t - this.lastAlert >= cfg.perclos_window_s) {
        this.lastAlert = t; events.push({ type: "FATIGUE_PERCLOS", perclos: rnd(perclos, 3), coverage: rnd(coverage, 3) });
      }
      return [out, events];
    }
  }

  // ------------------------------------------------------------------ speech mask (FR-2.2)
  class SpeechMask {
    constructor(cfg) {
      this.aperture = new TimeWindow(cfg.speech_window_s); this.jaw = new TimeWindow(cfg.speech_window_s);
      this.stdThresh = cfg.speech_aperture_std; this.openThresh = cfg.speech_open_delta;
      this.hold = 0.4; this.lastActive = null;
    }
    update(t, aperture, jaw, baseAperture, baseJaw) {
      this.aperture.add(t, aperture); this.jaw.add(t, jaw);
      const std = Math.max(this.aperture.std(), this.jaw.std());
      const openDelta = Math.max(aperture - baseAperture, jaw - baseJaw);
      let active = std > this.stdThresh || (openDelta > this.openThresh && std > this.stdThresh * 0.5);
      if (active) this.lastActive = t;
      else if (this.lastActive !== null && t - this.lastActive < this.hold) active = true;
      return { active, aperture_std: std, open_delta: openDelta };
    }
  }

  // ------------------------------------------------------------------ AU regression core
  function geometricEvidence(f, base, cfg) {
    const b = base.features, s = cfg.scales, d = {};
    for (const k of FEATURE_NAMES) d[k] = f[k] - b[k];
    const innerDrop = -(d.brow_inner_h_r + d.brow_inner_h_l) / 2;
    const gapDrop = -d.brow_gap;
    const outerRise = (d.brow_outer_h_r + d.brow_outer_h_l) / 2;
    const earBase = Math.max((b.ear_r + b.ear_l) / 2, 1e-6);
    const earDrop = 1 - (f.ear_r + f.ear_l) / 2 / earBase;
    const lidRise = -(d.lower_lid_r + d.lower_lid_l) / 2;
    return {
      au04: clip01((0.6 * innerDrop) / s.brow_height + (0.4 * gapDrop) / s.brow_gap),
      au01: clip01(-innerDrop / s.brow_height),
      au02: clip01(outerRise / s.outer_brow_height),
      au07: clip01((0.5 * earDrop) / s.eye_aperture_ratio + (0.5 * lidRise) / s.lower_lid_raise),
      au14: clip01((0.6 * d.lip_corner_depth) / s.lip_corner_depth + (0.4 * d.mouth_width) / s.mouth_width),
    };
  }

  function bsRel(bs, base, keys) {
    const vals = [];
    for (const k of keys) if (k in bs) { const rest = base[k] || 0; vals.push(clip01((bs[k] - rest) / Math.max(1 - rest, 1e-6))); }
    return vals.length ? mean(vals) : NaN;
  }

  function blendshapeEvidence(bs, base) {
    const rb = base.blendshapes || {};
    return {
      au04: bsRel(bs, rb, ["browDownLeft", "browDownRight"]),
      au01: bsRel(bs, rb, ["browInnerUp"]),
      au02: bsRel(bs, rb, ["browOuterUpLeft", "browOuterUpRight"]),
      au07: bsRel(bs, rb, ["eyeSquintLeft", "eyeSquintRight"]),
      au14: bsRel(bs, rb, ["mouthDimpleLeft", "mouthDimpleRight"]),
    };
  }

  class AURegressor {
    constructor(cfg) { this.cfg = cfg; this.ema = {}; AU_KEYS.forEach((k) => (this.ema[k] = new EMA(cfg.smoothing_tau_s))); this.heldAu7 = 0; }
    reset() { AU_KEYS.forEach((k) => this.ema[k].reset()); }
    update(t, feats, base, blendshapes, eyesClosed, speechActive) {
      const cfg = this.cfg, ev = geometricEvidence(feats, base, cfg);
      if (blendshapes && cfg.blendshape_weight > 0) {
        const bev = blendshapeEvidence(blendshapes, base), w = cfg.blendshape_weight;
        for (const k of Object.keys(bev)) if (!Number.isNaN(bev[k])) ev[k] = (1 - w) * ev[k] + w * bev[k];
      }
      if (eyesClosed) ev.au07 = this.heldAu7; else this.heldAu7 = ev.au07;
      if (speechActive) ev.au14 *= cfg.speech_lower_face_attenuation;
      const out = {};
      AU_KEYS.forEach((k) => (out[k] = clip01(this.ema[k].update(ev[k], t))));
      return out;
    }
  }

  // ------------------------------------------------------------------ cognitive metrics (FR-3.x)
  class CognitiveMetrics {
    constructor(cfg) {
      this.cfg = cfg; this.effort = new TimeWindow(cfg.mes_window_s);
      this.au4 = new TimeWindow(Math.max(cfg.cfi_rise_window_s, 0.2) + 0.05);
      this.spikeUntil = -1e9; this.lastCfiEvent = -1e9; this.browRaiseT = -1e9; this.surpriseUntil = -1e9; this.cfiPeak = 0;
    }
    rise(t, au4, window) {
      let lo = au4;
      for (const [ti, v] of this.au4.items) if (t - ti <= window && v < lo) lo = v;
      return au4 - lo;
    }
    update(t, au, blinkRate, baseBlinkRate, fusion) {
      const cfg = this.cfg, events = [];
      this.au4.add(t, au.au04);
      let suppression = null, effort;
      if (blinkRate !== null && baseBlinkRate) {
        suppression = clip01(1 - blinkRate / baseBlinkRate);
        effort = cfg.mes_au4_weight * au.au04 + (1 - cfg.mes_au4_weight) * suppression;
      } else effort = au.au04;
      this.effort.add(t, effort);
      const mes = 100 * (this.effort.mean() || 0);

      const rise = this.rise(t, au.au04, cfg.cfi_rise_window_s);
      const sacc = cfg.cfi_saccade_rate_hz ? clip01(fusion.saccade_rate_hz / cfg.cfi_saccade_rate_hz) : 0;
      const co = Math.max(clip01(au.au14 / cfg.cfi_au14_threshold), sacc);
      let cfi = clip01(au.au04 * (0.6 + 0.4 * co) + 0.15 * clip01(rise / cfg.cfi_rise_min));
      const taskRecent = fusion.seconds_since_task_complete !== null && fusion.seconds_since_task_complete <= cfg.cfi_task_completion_window_s;
      if (taskRecent) cfi *= 0.5;

      if (au.au04 > cfg.cfi_au4_threshold && rise >= cfg.cfi_rise_min) { this.spikeUntil = t + 0.3; this.cfiPeak = 0; }
      if (t <= this.spikeUntil) {
        this.cfiPeak = Math.max(this.cfiPeak, cfi);
        const coAu14 = au.au14 >= cfg.cfi_au14_threshold, coSacc = fusion.saccade_rate_hz >= cfg.cfi_saccade_rate_hz;
        if ((coAu14 || coSacc) && !taskRecent && t - this.lastCfiEvent >= cfg.cfi_event_refractory_s) {
          this.lastCfiEvent = t; this.spikeUntil = -1e9;
          events.push({ type: "CFI_EVENT", cfi: rnd(this.cfiPeak, 3), au04: rnd(au.au04, 3), trigger: coAu14 ? "AU14" : "SACCADES", aoi: fusion.active_aoi });
        }
      }

      if ((au.au01 + au.au02) / 2 > cfg.surprise_brow_threshold) this.browRaiseT = t;
      const surge = au.au04 >= cfg.surprise_au4_surge && this.rise(t, au.au04, 0.2) >= 0.2;
      if (surge && t - this.browRaiseT > 0 && t - this.browRaiseT <= cfg.surprise_window_s && t > this.surpriseUntil) {
        this.surpriseUntil = t + cfg.surprise_latch_s; this.browRaiseT = -1e9;
        events.push({ type: "AUTOMATION_SURPRISE", au04: rnd(au.au04, 3), aoi: fusion.active_aoi });
      }
      return { mental_effort_score: rnd(mes, 2), cognitive_friction_index: rnd(cfi, 3),
        automation_surprise_flag: t <= this.surpriseUntil, blink_suppression: suppression, events };
    }
  }

  // ------------------------------------------------------------------ fusion helpers (FR-4.3)
  class CompoundRiskTracker {
    constructor(rearm) { this.rearm = rearm || 1; this.active = false; this.clearSince = null; }
    update(t, cond) {
      if (cond) { this.clearSince = null; if (!this.active) { this.active = true; return true; } return false; }
      if (this.active) { if (this.clearSince === null) this.clearSince = t; else if (t - this.clearSince >= this.rearm) this.active = false; }
      return false;
    }
  }

  function correlate(ctx, au04, au07, cfi, mes, surprise, microsleep, cfg, t, tracker) {
    const events = [];
    const neckHigh = ctx.rula_neck_score !== null && ctx.rula_neck_score >= cfg.rula_neck_threshold;
    const trunkHigh = ctx.rula_trunk_score !== null && ctx.rula_trunk_score >= cfg.rula_trunk_threshold;
    const posture = neckHigh || trunkHigh, visual = au07 >= cfg.compound_au7_threshold, cog = au04 >= cfg.compound_au4_threshold;
    const compound = posture && (visual || cog);
    if (tracker ? tracker.update(t, compound) : compound) {
      events.push({ type: "COMPOUND_POSTURE_RISK", rula_neck: ctx.rula_neck_score, rula_trunk: ctx.rula_trunk_score,
        au07: rnd(au07, 3), au04: rnd(au04, 3), aoi: ctx.active_aoi });
    }
    let insight = null;
    if (microsleep) insight = "FATIGUE_MICROSLEEP_RISK";
    else if (surprise) insight = "AUTOMATION_SURPRISE";
    else if (posture && visual) insight = "POSTURE_DRIVEN_VISUAL_COMPENSATION";
    else if (cfi >= cfg.cfi_marker_threshold && ctx.active_aoi) insight = "HIGH_COGNITIVE_STRAIN_ON_COMPLEX_WIDGET";
    else if (cfi >= cfg.cfi_marker_threshold) insight = "HIGH_COGNITIVE_FRICTION";
    else if (visual && !posture) insight = "VISUAL_STRAIN_DISPLAY_LEGIBILITY";
    else if (posture && cog) insight = "POSTURAL_LOAD_WITH_COGNITIVE_STRAIN";
    else if (mes >= 60) insight = "SUSTAINED_MENTAL_EFFORT";
    return [insight, events];
  }

  // ------------------------------------------------------------------ engine
  const PROVISIONAL_FRAMES = 30;
  const poseFactor = (v, limit, falloff) => { const ex = Math.abs(v) - limit; return ex <= 0 ? 1 : clip01(1 - ex / falloff); };

  class CogSenseEngine {
    constructor(overrides, baseline) {
      this.cfg = makeConfig(overrides);
      this.baseline = baseline || null;
      this.au = new AURegressor(this.cfg);
      this.blink = new BlinkDetector(this.cfg);
      this.speech = new SpeechMask(this.cfg);
      this.metrics = new CognitiveMetrics(this.cfg);
      this.fatigue = new FatigueTracker(this.cfg);
      this.calibrator = new BaselineCalibrator(this.cfg.calibration_seconds, this.cfg.calibration_min_frames);
      this.compound = new CompoundRiskTracker();
      this.provisional = []; this.provisionalBs = []; this.prevPoints = null;
      this.context = { active_aoi: null, gaze_x: null, gaze_y: null, saccade_rate_hz: 0, rula_grand_score: null,
        rula_neck_score: null, rula_trunk_score: null, seconds_since_task_complete: null, mission_phase: null };
    }
    /** Merge external context (gaze/AOI/RULA) for the next frames. */
    setContext(patch) { Object.assign(this.context, patch); }
    startCalibration(t) { this.calibrator.start(t); }
    get calibrationStatus() {
      if (this.calibrator.state === "COLLECTING") return "CALIBRATING";
      if (!this.baseline) return "UNCALIBRATED";
      return this.baseline.is_default ? "PROVISIONAL" : "CALIBRATED";
    }
    baselineBlinkRate() { return (this.baseline && this.baseline.blink_rate_per_min) || this.cfg.default_blink_rate_per_min; }

    /** obs: {timestamp_s, landmarks|null, width, height, frame_id, timestamp_utc_ms?, blendshapes?, luma_mean?} */
    process(obs) {
      const t0 = nowMs(), cfg = this.cfg;
      const utc = obs.timestamp_utc_ms !== undefined ? obs.timestamp_utc_ms : Date.now();
      const lm = obs.landmarks;
      if (!lm || lm.length < LM.MIN_LANDMARKS) return this.finish(this.lostPayload(obs, utc, "NO_FACE", null, 0), t0);

      const face = normalizeFace(lm, obs.width, obs.height);
      const feats = extractFeatures(face.points);
      const poseOk = Math.abs(face.yaw) <= cfg.yaw_limit_deg && Math.abs(face.pitch) <= cfg.pitch_limit_deg && Math.abs(face.roll) <= cfg.roll_limit_deg;
      let conf = poseFactor(face.yaw, cfg.yaw_limit_deg, cfg.pose_falloff_deg) * poseFactor(face.pitch, cfg.pitch_limit_deg, cfg.pose_falloff_deg)
        * poseFactor(face.roll, cfg.roll_limit_deg, cfg.pose_falloff_deg);
      if (face.iod < 2 * cfg.min_iod_px) conf *= clip01(face.iod / (2 * cfg.min_iod_px));
      if (this.prevPoints) {
        let s = 0;
        for (let i = 0; i < LM.MIN_LANDMARKS; i++) {
          const a = face.points[i], b = this.prevPoints[i];
          s += Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
        }
        conf *= clip01(1 - (s / LM.MIN_LANDMARKS - 0.08) / 0.25);
      }
      this.prevPoints = face.points.slice(0, LM.MIN_LANDMARKS);
      if (feats.brow_inner_h_r <= 0 || feats.brow_inner_h_l <= 0) conf *= 0.3;
      if (obs.luma_mean !== undefined && obs.luma_mean !== null && !(obs.luma_mean >= 25 && obs.luma_mean <= 235)) conf *= 0.6;

      if (!this.baseline) {
        if (conf >= cfg.locked_confidence) {
          this.provisional.push(FEATURE_NAMES.map((k) => feats[k]));
          if (obs.blendshapes) this.provisionalBs.push(obs.blendshapes);
        }
        if (this.provisional.length >= PROVISIONAL_FRAMES) {
          const features = {}, blendshapes = {};
          FEATURE_NAMES.forEach((k, j) => (features[k] = median(this.provisional.map((r) => r[j]))));
          for (const k of BLENDSHAPE_KEYS) {
            const vals = this.provisionalBs.filter((b) => k in b).map((b) => b[k]);
            if (vals.length) blendshapes[k] = median(vals);
          }
          this.baseline = { features, feature_mad: {}, blendshapes, blink_rate_per_min: null, frames: this.provisional.length, is_default: true };
        } else return this.finish(this.lostPayload(obs, utc, "ACQUIRING", face, conf), t0);
      }

      const base = this.baseline, T = obs.timestamp_s;
      const earBase = Math.max((base.features.ear_r + base.features.ear_l) / 2, 1e-6);
      const blink = this.blink.update(T, (feats.ear_r + feats.ear_l) / 2 / earBase);
      const speech = this.speech.update(T, feats.mouth_aperture, feats.jaw_drop, base.features.mouth_aperture, base.features.jaw_drop);
      if (speech.active) conf *= cfg.speech_confidence_factor;
      if (!blink.closed) {
        const asym = Math.abs(feats.ear_r - feats.ear_l) / Math.max(feats.ear_r, feats.ear_l, 1e-6);
        if (asym > 0.6) conf *= 0.5;
      }

      if (this.calibrator.state === "COLLECTING") {
        const nb = this.calibrator.add(T, feats, obs.blendshapes || null, blink.closed, blink.onset, conf >= cfg.locked_confidence && !speech.active);
        if (nb) { this.baseline = nb; this.au.reset(); }
      }

      if (conf < cfg.lost_confidence) { this.au.reset(); return this.finish(this.lostPayload(obs, utc, "LOW_CONFIDENCE", face, conf), t0); }

      const au = this.au.update(T, feats, this.baseline, obs.blendshapes || null, blink.closed, speech.active);
      const [fat, fatEvents] = this.fatigue.update(T, (feats.ear_r + feats.ear_l) / 2 / earBase, blink);
      const ctx = this.context;
      const m = this.metrics.update(T, au, blink.rate_per_min, this.baselineBlinkRate(),
        { saccade_rate_hz: ctx.saccade_rate_hz, seconds_since_task_complete: ctx.seconds_since_task_complete, active_aoi: ctx.active_aoi });
      const [insight, fusionEvents] = correlate(ctx, au.au04, au.au07, m.cognitive_friction_index, m.mental_effort_score,
        m.automation_surprise_flag, blink.microsleep, cfg, T, this.compound);
      const events = m.events.concat(fusionEvents, fatEvents);
      if (blink.microsleep && blink.closure_ms - 1000 / cfg.target_hz < cfg.microsleep_ms) events.push({ type: "MICROSLEEP", closure_ms: rnd(blink.closure_ms, 1) });

      return this.finish({
        timestamp_utc_ms: utc, frame_id: obs.frame_id || 0,
        tracking_status: conf >= cfg.locked_confidence ? "LOCKED" : "DEGRADED", confidence: rnd(conf, 3),
        head_pose: this.pose(face, poseOk),
        action_units: {
          au04_brow_lowerer: rnd(au.au04, 3), au07_lid_tightener: rnd(au.au07, 3), au01_inner_brow_raiser: rnd(au.au01, 3),
          au02_outer_brow_raiser: rnd(au.au02, 3), au14_dimpler: rnd(au.au14, 3), au45_blink_state: blink.closed ? 1 : 0,
          au43_eyes_closed: !!blink.microsleep,
        },
        blink: { rate_per_min: blink.rate_per_min === null ? null : rnd(blink.rate_per_min, 1),
          last_duration_ms: blink.last_blink_ms === null ? null : rnd(blink.last_blink_ms, 1), closure_ms: rnd(blink.closure_ms, 1) },
        cognitive_metrics: { mental_effort_score: m.mental_effort_score, cognitive_friction_index: m.cognitive_friction_index,
          automation_surprise_flag: m.automation_surprise_flag, speech_interference_detected: speech.active },
        fatigue: fat,
        fusion_context: this.fusionOut(ctx, insight),
        calibration: { status: this.calibrationStatus, progress: rnd(this.calibrator.progress(T), 3) },
        events,
      }, t0);
    }

    pose(face, ok) { return { yaw_deg: rnd(face.yaw, 2), pitch_deg: rnd(face.pitch, 2), roll_deg: rnd(face.roll, 2), within_operating_range: ok }; }

    fusionOut(ctx, insight) {
      return { active_aoi: ctx.active_aoi, gaze_px: ctx.gaze_x === null ? null : [rnd(ctx.gaze_x, 1), rnd(ctx.gaze_y, 1)],
        saccade_rate_hz: ctx.saccade_rate_hz, rula_grand_score: ctx.rula_grand_score, rula_neck_score: ctx.rula_neck_score,
        rula_trunk_score: ctx.rula_trunk_score, mission_phase: ctx.mission_phase, pupil: null, correlated_insight: insight };
    }

    /** Graceful fallback: explicit low-confidence flag, never stale or hallucinated AUs. */
    lostPayload(obs, utc, reason, face, conf) {
      this.fatigue.markGap(); // a gap must never count as open or closed time
      if (!face) { this.blink.resetGap(); this.au.reset(); this.prevPoints = null; }
      return {
        timestamp_utc_ms: utc, frame_id: obs.frame_id || 0,
        tracking_status: reason === "ACQUIRING" ? "ACQUIRING" : "LOST", tracking_loss_reason: reason,
        confidence: rnd(reason !== "ACQUIRING" ? Math.min(conf, this.cfg.lost_confidence - 1e-3) : conf, 3),
        head_pose: face ? this.pose(face, false) : null, action_units: null, blink: null, cognitive_metrics: null, fatigue: null,
        fusion_context: this.fusionOut(this.context, null),
        calibration: { status: this.calibrationStatus, progress: rnd(this.calibrator.progress(obs.timestamp_s), 3) }, events: [],
      };
    }

    finish(payload, t0) { payload.processing_ms = rnd(nowMs() - t0, 3); return payload; }
  }

  return { CogSenseEngine, makeConfig, DEFAULTS, normalizeFace, extractFeatures, FEATURE_NAMES, LM };
});
