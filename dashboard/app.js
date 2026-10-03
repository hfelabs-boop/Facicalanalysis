/* CogSense HFE Analytics Console — live HUD, synchronized timeline, gaze heatmap, playback. */
(() => {
  "use strict";

  const SCREEN_W = 1920, SCREEN_H = 1080;
  const MAX_BUFFER = 250000;
  let aois = [
    { name: "TACTICAL_RADAR_WIDGET_PRIMARY", x: 40, y: 60, w: 1100, h: 700 },
    { name: "TRACK_TABLE", x: 1180, y: 60, w: 700, h: 420 },
    { name: "ASSET_STATUS_PANEL", x: 1180, y: 500, w: 700, h: 260 },
    { name: "ENGAGEMENT_MENU_NESTED", x: 40, y: 800, w: 700, h: 240 },
    { name: "COMMS_LOG", x: 780, y: 800, w: 1100, h: 240 },
  ];
  const AU_DEFS = [
    ["au04_brow_lowerer", "AU4 Brow lowerer"],
    ["au07_lid_tightener", "AU7 Lid tightener"],
    ["au01_inner_brow_raiser", "AU1 Inner brow raiser"],
    ["au02_outer_brow_raiser", "AU2 Outer brow raiser"],
    ["au14_dimpler", "AU14 Dimpler"],
    ["au45_blink_state", "AU45 Blink"],
  ];

  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  const state = {
    mode: "live",
    buffer: [],
    events: [],
    cursor: -1,
    follow: true,
    ws: null,
    lastMesh: null,
    recording: false,
    recStart: null,
    playing: false,
    playTimer: null,
    dirtyTimeline: true,
  };

  // ---------------------------------------------------------------- AU bars
  const auBars = $("auBars");
  const auEls = {};
  for (const [key, label] of AU_DEFS) {
    const row = document.createElement("div");
    row.className = "au-row";
    row.innerHTML = `<span class="name">${label}</span><div class="track"><div class="fill"></div></div><span class="val">--</span>`;
    auBars.appendChild(row);
    auEls[key] = { fill: row.querySelector(".fill"), val: row.querySelector(".val") };
  }

  // ---------------------------------------------------------------- helpers
  const fmtT = (s) => {
    s = Math.max(0, Math.floor(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    const mm = String(m).padStart(2, "0"), ss = String(sec).padStart(2, "0");
    return h ? `${String(h).padStart(2, "0")}:${mm}:${ss}` : `${mm}:${ss}`;
  };
  const hms = (s) => {
    s = Math.max(0, Math.floor(s));
    return [Math.floor(s / 3600), Math.floor((s % 3600) / 60), s % 60].map((v) => String(v).padStart(2, "0")).join(":");
  };
  const t0 = () => (state.buffer.length ? state.buffer[0].timestamp_utc_ms : 0);
  const relT = (p) => (p.timestamp_utc_ms - t0()) / 1000;
  const current = () => state.buffer[state.cursor] || null;

  function ingest(p) {
    if (p.hud_mesh) { state.lastMesh = p.hud_mesh; delete p.hud_mesh; }
    if (typeof p.recording === "boolean") {
      if (p.recording && !state.recording) state.recStart = Date.now();
      state.recording = p.recording;
    }
    state.buffer.push(p);
    if (state.buffer.length > MAX_BUFFER) {
      state.buffer.splice(0, state.buffer.length - MAX_BUFFER);
      state.events = state.events.filter((e) => e.ms >= state.buffer[0].timestamp_utc_ms);
    }
    for (const ev of p.events || []) addEvent(ev, p);
    if (state.follow) state.cursor = state.buffer.length - 1;
    state.dirtyTimeline = true;
  }

  function addEvent(ev, p) {
    const e = { ...ev, ms: p.timestamp_utc_ms, idx: state.buffer.length - 1 };
    state.events.push(e);
    const li = document.createElement("li");
    li.className = ev.type;
    li.dataset.ms = e.ms;
    li.innerHTML = `<span class="t">[${fmtT(relT(p))}]</span>${describe(ev, p)}`;
    li.onclick = () => seekMs(e.ms);
    const list = $("findings");
    list.prepend(li);
    while (list.children.length > 300) list.lastChild.remove();
  }

  function describe(ev, p) {
    const aoi = ev.aoi ? ` → AOI "${ev.aoi}"` : "";
    switch (ev.type) {
      case "CFI_EVENT": return `Friction spike (CFI ${ev.cfi.toFixed(2)}, ${ev.trigger})${aoi}`;
      case "AUTOMATION_SURPRISE": return `Automation surprise (AU1+2 → AU4 ${ev.au04.toFixed(2)})${aoi}`;
      case "COMPOUND_POSTURE_RISK":
        return `Posture risk: RULA neck ${ev.rula_neck ?? "-"} / trunk ${ev.rula_trunk ?? "-"} + AU7 ${ev.au07.toFixed(2)}, AU4 ${ev.au04.toFixed(2)}${aoi}`;
      case "MICROSLEEP": return `Prolonged eye closure (AU43) ${Math.round(ev.closure_ms)} ms`;
      default: return ev.type;
    }
  }

  function seekMs(ms) {
    let lo = 0, hi = state.buffer.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (state.buffer[mid].timestamp_utc_ms < ms) lo = mid + 1; else hi = mid; }
    setCursor(lo, false);
  }

  function setCursor(i, follow) {
    state.cursor = Math.max(0, Math.min(i, state.buffer.length - 1));
    state.follow = follow;
    state.dirtyTimeline = true;
    syncVideos();
  }

  // ---------------------------------------------------------------- live HUD
  function renderHud() {
    const p = current();
    if (!p) return;
    const au = p.action_units, cm = p.cognitive_metrics, fc = p.fusion_context || {}, hp = p.head_pose;
    const st = $("trackStatus");
    st.textContent = p.tracking_status;
    st.style.color = p.tracking_status === "LOCKED" ? css("--ok") : p.tracking_status === "DEGRADED" ? css("--warn") : css("--bad");
    $("trackConf").textContent = `${Math.round((p.confidence || 0) * 100)}%`;
    $("calStatus").textContent = p.calibration
      ? p.calibration.status + (p.calibration.status === "CALIBRATING" ? ` ${Math.round(p.calibration.progress * 100)}%` : "")
      : "--";
    $("yaw").textContent = hp ? hp.yaw_deg.toFixed(1) : "--";
    $("pitch").textContent = hp ? hp.pitch_deg.toFixed(1) : "--";
    $("roll").textContent = hp ? hp.roll_deg.toFixed(1) : "--";
    $("blinkRate").textContent = p.blink && p.blink.rate_per_min != null ? p.blink.rate_per_min.toFixed(0) : "--";
    $("speech").hidden = !(cm && cm.speech_interference_detected);

    for (const [key] of AU_DEFS) {
      const v = au ? Number(au[key]) : null;
      auEls[key].fill.style.width = v == null ? "0" : `${Math.min(v, 1) * 100}%`;
      auEls[key].fill.style.background = v != null && v >= 0.65 ? css("--cfi") : v >= 0.4 ? css("--warn") : css("--accent");
      auEls[key].val.textContent = v == null ? "--" : key === "au45_blink_state" ? (v ? "CLOSED" : "open") : v.toFixed(2);
    }
    $("mesVal").textContent = cm ? cm.mental_effort_score.toFixed(1) : "--";
    $("mesBar").style.width = cm ? `${cm.mental_effort_score}%` : "0";
    $("cfiVal").textContent = cm ? cm.cognitive_friction_index.toFixed(2) : "--";
    $("cfiBar").style.width = cm ? `${cm.cognitive_friction_index * 100}%` : "0";
    $("aoi").textContent = fc.active_aoi || "--";
    $("rula").textContent = fc.rula_grand_score ?? "--";
    $("neck").textContent = fc.rula_neck_score ?? "--";
    $("insight").textContent = fc.correlated_insight || (au ? "No correlated finding" : `Tracking unavailable (${p.tracking_loss_reason || p.tracking_status}) — metrics withheld`);

    drawFace(p);
    drawGaze();
    $("rec").textContent = state.recording && state.recStart ? `REC ${hms((Date.now() - state.recStart) / 1000)}` : "REC --:--:--";
    $("rec").classList.toggle("recording", state.recording);
    $("record").textContent = state.recording ? "Stop rec" : "Record";
    $("cursorTime").textContent = `cursor ${fmtT(relT(p))}${fc.mission_phase ? " · " + fc.mission_phase : ""}`;
  }

  // ---------------------------------------------------------------- face canvas
  const mesh = $("mesh"), mctx = mesh.getContext("2d");
  function drawFace(p) {
    const W = mesh.width, H = mesh.height;
    mctx.clearRect(0, 0, W, H);
    const au = p.action_units;
    if (state.mode === "live" && state.lastMesh && au) {
      const pts = state.lastMesh;
      let minx = 1, maxx = 0, miny = 1, maxy = 0;
      for (const [x, y] of pts) { minx = Math.min(minx, x); maxx = Math.max(maxx, x); miny = Math.min(miny, y); maxy = Math.max(maxy, y); }
      const s = Math.min((W * 0.8) / (maxx - minx || 1), (H * 0.85) / (maxy - miny || 1));
      const ox = W / 2 - ((minx + maxx) / 2) * s, oy = H / 2 - ((miny + maxy) / 2) * s;
      mctx.fillStyle = p.tracking_status === "LOCKED" ? "#3fcf8eaa" : "#f0a23baa";
      for (const [x, y] of pts) mctx.fillRect(x * s + ox, y * s + oy, 1.6, 1.6);
    } else {
      drawSchematic(au, W, H);
    }
    mctx.font = "12px ui-monospace, monospace";
    if (au) {
      mctx.fillStyle = au.au04_brow_lowerer >= 0.65 ? css("--cfi") : "#d6dde6";
      mctx.fillText(`[AU4: ${au.au04_brow_lowerer.toFixed(2)}]`, W - 120, 22);
      mctx.fillStyle = au.au07_lid_tightener >= 0.4 ? css("--warn") : "#d6dde6";
      mctx.fillText(`[AU7: ${au.au07_lid_tightener.toFixed(2)}]`, W - 120, 40);
    } else {
      mctx.fillStyle = css("--bad");
      mctx.fillText(`TRACKING ${p.tracking_status} — conf ${(p.confidence || 0).toFixed(2)}`, 12, 22);
    }
  }

  function drawSchematic(au, W, H) {
    const cx = W / 2, cy = H / 2 + 10, k = H / 300;
    mctx.strokeStyle = au ? "#3fcf8e" : "#55606e";
    mctx.lineWidth = 2;
    mctx.beginPath(); mctx.ellipse(cx, cy, 95 * k, 125 * k, 0, 0, Math.PI * 2); mctx.stroke();
    if (!au) return;
    const a4 = au.au04_brow_lowerer, a1 = au.au01_inner_brow_raiser, a2 = au.au02_outer_brow_raiser;
    const a7 = au.au07_lid_tightener, blink = au.au45_blink_state, a14 = au.au14_dimpler;
    for (const side of [-1, 1]) {
      const ex = cx + side * 40 * k, ey = cy - 25 * k;
      const inY = ey - 30 * k + 14 * k * a4 - 14 * k * a1, outY = ey - 30 * k - 14 * k * a2 + 4 * k * a4;
      const inX = cx + side * (15 - 6 * a4) * k;
      mctx.beginPath(); mctx.moveTo(inX, inY); mctx.quadraticCurveTo(ex, Math.min(inY, outY) - 6 * k, cx + side * 68 * k, outY); mctx.stroke();
      const open = blink ? 0.6 : 9 * (1 - 0.6 * a7);
      mctx.beginPath(); mctx.ellipse(ex, ey, 18 * k, Math.max(open, 0.6) * k, 0, 0, Math.PI * 2); mctx.stroke();
    }
    const my = cy + 60 * k, mw = (34 + 6 * a14) * k;
    mctx.beginPath(); mctx.moveTo(cx - mw, my - 3 * k * a14); mctx.quadraticCurveTo(cx, my + 4 * k, cx + mw, my - 3 * k * a14); mctx.stroke();
  }

  // ---------------------------------------------------------------- gaze canvas
  const gaze = $("gaze"), gctx = gaze.getContext("2d");
  const heat = document.createElement("canvas");
  heat.width = 192; heat.height = 108;
  const hctx = heat.getContext("2d");
  function drawGaze() {
    const W = gaze.width, H = gaze.height, sx = W / SCREEN_W, sy = H / SCREEN_H;
    gctx.clearRect(0, 0, W, H);
    gctx.lineWidth = 1;
    const cur = current();
    const active = cur && cur.fusion_context ? cur.fusion_context.active_aoi : null;
    for (const a of aois) {
      gctx.strokeStyle = a.name === active ? css("--warn") : "#3a4656";
      gctx.strokeRect(a.x * sx, a.y * sy, a.w * sx, a.h * sy);
      gctx.fillStyle = "#7d8a99";
      gctx.font = "9px ui-monospace, monospace";
      gctx.fillText(a.name, a.x * sx + 3, a.y * sy + 10);
    }
    // heatmap over the last 60 s up to the cursor, weighted by CFI
    const end = state.cursor, buf = state.buffer;
    if (end < 0) return;
    const endMs = buf[end].timestamp_utc_ms;
    hctx.clearRect(0, 0, heat.width, heat.height);
    const step = Math.max(1, Math.floor((end + 1) / 4000));
    for (let i = end; i >= 0 && endMs - buf[i].timestamp_utc_ms < 60000; i -= step) {
      const fc = buf[i].fusion_context;
      if (!fc || !fc.gaze_px) continue;
      const x = (fc.gaze_px[0] / SCREEN_W) * heat.width, y = (fc.gaze_px[1] / SCREEN_H) * heat.height;
      const cfi = buf[i].cognitive_metrics ? buf[i].cognitive_metrics.cognitive_friction_index : 0;
      const g = hctx.createRadialGradient(x, y, 0, x, y, 7);
      g.addColorStop(0, `rgba(255,${Math.round(200 - 180 * cfi)},60,0.06)`);
      g.addColorStop(1, "rgba(255,120,60,0)");
      hctx.fillStyle = g;
      hctx.fillRect(x - 7, y - 7, 14, 14);
    }
    gctx.imageSmoothingEnabled = true;
    gctx.drawImage(heat, 0, 0, W, H);
    // scan path over the last 3 s
    gctx.strokeStyle = "#4ea1ffcc";
    gctx.beginPath();
    let first = true;
    for (let i = Math.max(0, end - 400); i <= end; i++) {
      const fc = buf[i].fusion_context;
      if (!fc || !fc.gaze_px || endMs - buf[i].timestamp_utc_ms > 3000) continue;
      const x = fc.gaze_px[0] * sx, y = fc.gaze_px[1] * sy;
      if (first) { gctx.moveTo(x, y); first = false; } else gctx.lineTo(x, y);
    }
    gctx.stroke();
    if (cur && cur.fusion_context && cur.fusion_context.gaze_px) {
      gctx.fillStyle = "#fff";
      gctx.beginPath(); gctx.arc(cur.fusion_context.gaze_px[0] * sx, cur.fusion_context.gaze_px[1] * sy, 3.5, 0, Math.PI * 2); gctx.fill();
    }
  }

  // ---------------------------------------------------------------- timeline
  const tl = $("timeline"), tctx = tl.getContext("2d");
  function drawTimeline() {
    const dpr = window.devicePixelRatio || 1;
    const cw = tl.clientWidth, ch = tl.clientHeight;
    if (tl.width !== cw * dpr) { tl.width = cw * dpr; tl.height = ch * dpr; }
    tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    tctx.clearRect(0, 0, cw, ch);
    const buf = state.buffer;
    const L = 46, R = 10, top = 14, mid = ch * 0.58, bottom = ch - 22;
    const pw = cw - L - R;
    tctx.font = "11px ui-sans-serif, system-ui";
    tctx.fillStyle = css("--muted");
    tctx.strokeStyle = "#2a3441";
    tctx.lineWidth = 1;
    tctx.beginPath(); tctx.moveTo(L, mid); tctx.lineTo(L + pw, mid); tctx.moveTo(L, bottom); tctx.lineTo(L + pw, bottom); tctx.stroke();
    tctx.fillText("MES 100", 2, top + 8); tctx.fillText("0", L - 12, mid);
    tctx.fillText("RULA 7", 2, mid + 16); tctx.fillText("1", L - 12, bottom);
    if (buf.length < 2) return;
    const tStart = buf[0].timestamp_utc_ms, tEnd = buf[buf.length - 1].timestamp_utc_ms, span = Math.max(tEnd - tStart, 1);
    const X = (ms) => L + ((ms - tStart) / span) * pw;

    // phase bands
    let phase = null, phaseStart = tStart;
    const bands = [];
    const n = buf.length, step = Math.max(1, Math.floor(n / (pw * 2)));
    for (let i = 0; i < n; i += step) {
      const ph = buf[i].fusion_context ? buf[i].fusion_context.mission_phase : null;
      if (ph !== phase) { if (phase) bands.push([phase, phaseStart, buf[i].timestamp_utc_ms]); phase = ph; phaseStart = buf[i].timestamp_utc_ms; }
    }
    if (phase) bands.push([phase, phaseStart, tEnd]);
    bands.forEach(([name, a, b], j) => {
      tctx.fillStyle = j % 2 ? "#ffffff06" : "#ffffff0d";
      tctx.fillRect(X(a), top, X(b) - X(a), bottom - top);
      tctx.fillStyle = "#7d8a99";
      tctx.save(); tctx.beginPath(); tctx.rect(X(a), 0, X(b) - X(a), top + 2); tctx.clip();
      tctx.fillText(name, X(a) + 3, top - 2); tctx.restore();
    });

    // MES line (bucketed max/avg to keep it fast)
    const plot = (getter, y0, y1, vmin, vmax, color, stepped) => {
      tctx.strokeStyle = color; tctx.lineWidth = 1.5; tctx.beginPath();
      let started = false, prevY = null;
      for (let i = 0; i < n; i += step) {
        const v = getter(buf[i]);
        if (v == null) { started = false; continue; }
        const x = X(buf[i].timestamp_utc_ms), y = y1 - ((v - vmin) / (vmax - vmin)) * (y1 - y0);
        if (!started) { tctx.moveTo(x, y); started = true; }
        else { if (stepped && prevY != null) tctx.lineTo(x, prevY); tctx.lineTo(x, y); }
        prevY = y;
      }
      tctx.stroke();
    };
    plot((p) => (p.cognitive_metrics ? p.cognitive_metrics.mental_effort_score : null), top, mid, 0, 100, css("--mes"), false);
    plot((p) => (p.fusion_context && p.fusion_context.rula_grand_score != null ? p.fusion_context.rula_grand_score : null), mid + 6, bottom, 1, 7, css("--rula"), true);

    // event markers
    const colors = { CFI_EVENT: css("--cfi"), AUTOMATION_SURPRISE: css("--sur"), COMPOUND_POSTURE_RISK: css("--cmp"), MICROSLEEP: css("--warn") };
    for (const e of state.events) {
      if (e.ms < tStart) continue;
      const x = X(e.ms);
      tctx.strokeStyle = colors[e.type] || "#fff";
      tctx.lineWidth = e.type === "CFI_EVENT" ? 2 : 1;
      tctx.beginPath();
      if (e.type === "COMPOUND_POSTURE_RISK") { tctx.moveTo(x, mid + 4); tctx.lineTo(x, bottom); }
      else { tctx.moveTo(x, top); tctx.lineTo(x, mid); }
      tctx.stroke();
    }

    // axis labels
    tctx.fillStyle = css("--muted");
    for (let k = 0; k <= 4; k++) {
      const ms = tStart + (span * k) / 4;
      const label = fmtT((ms - tStart) / 1000);
      tctx.fillText(label, Math.min(X(ms) - (k ? 14 : 0), cw - 40), ch - 6);
    }
    // cursor
    const cur = current();
    if (cur) {
      const x = X(cur.timestamp_utc_ms);
      tctx.strokeStyle = "#ffffffaa"; tctx.lineWidth = 1;
      tctx.beginPath(); tctx.moveTo(x, top); tctx.lineTo(x, bottom); tctx.stroke();
    }
    const scrub = $("scrub");
    scrub.max = String(n - 1);
    scrub.value = String(state.cursor);
  }

  $("scrub").addEventListener("input", (e) => {
    const i = Number(e.target.value);
    setCursor(i, state.mode === "live" && i >= state.buffer.length - 2);
  });
  tl.addEventListener("click", (e) => {
    if (state.buffer.length < 2) return;
    const r = tl.getBoundingClientRect(), L = 46, pw = r.width - L - 10;
    const frac = Math.min(Math.max((e.clientX - r.left - L) / pw, 0), 1);
    const tStart = state.buffer[0].timestamp_utc_ms, tEnd = state.buffer[state.buffer.length - 1].timestamp_utc_ms;
    seekMs(tStart + frac * (tEnd - tStart));
  });

  // ---------------------------------------------------------------- WebSocket
  function connect() {
    if (state.ws) { state.ws.close(); state.ws = null; return; }
    const ws = new WebSocket($("wsUrl").value);
    state.ws = ws;
    ws.onopen = () => { $("conn").textContent = "LIVE"; $("conn").className = "pill live"; $("connect").textContent = "Disconnect"; };
    ws.onclose = () => { $("conn").textContent = "OFFLINE"; $("conn").className = "pill off"; $("connect").textContent = "Connect"; state.ws = null; };
    ws.onmessage = (m) => { if (state.mode === "live") { try { ingest(JSON.parse(m.data)); } catch (err) { console.warn(err); } } };
  }
  const send = (obj) => { if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(obj)); };
  $("connect").onclick = connect;
  $("calibrate").onclick = () => send({ type: "control", cmd: "calibrate" });
  $("record").onclick = () => send({ type: "control", cmd: state.recording ? "record_stop" : "record_start" });
  $("sendAoi").onclick = () => send({ type: "aoi_layout", aois });

  // ---------------------------------------------------------------- modes & playback
  function setMode(mode) {
    state.mode = mode;
    $("modeLive").classList.toggle("on", mode === "live");
    $("modePlayback").classList.toggle("on", mode === "playback");
    $("playbackBar").hidden = mode !== "playback";
    for (const el of [$("operatorVideo"), $("screenVideo")]) el.hidden = mode !== "playback" || !el.src;
    if (mode === "live") { stopPlay(); resetBuffer(); state.follow = true; }
  }
  function resetBuffer() {
    state.buffer = []; state.events = []; state.cursor = -1; $("findings").innerHTML = ""; state.dirtyTimeline = true;
  }
  $("modeLive").onclick = () => setMode("live");
  $("modePlayback").onclick = () => setMode("playback");

  $("fileSession").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    resetBuffer();
    const text = await f.text();
    const rows = text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
    rows.sort((a, b) => a.timestamp_utc_ms - b.timestamp_utc_ms);
    state.follow = false;
    for (const r of rows) ingest(r);
    setCursor(0, false);
    $("playbackInfo").textContent = `${rows.length} frames · ${fmtT(relT(rows[rows.length - 1]))} · ${state.events.length} events`;
  });

  function attachVideo(input, el) {
    input.addEventListener("change", (e) => {
      const f = e.target.files[0];
      if (!f) return;
      el.src = URL.createObjectURL(f);
      el.hidden = false;
      syncVideos();
    });
  }
  attachVideo($("fileOperator"), $("operatorVideo"));
  attachVideo($("fileScreen"), $("screenVideo"));

  function syncVideos() {
    const p = current();
    if (!p || state.mode !== "playback") return;
    const t = relT(p) + Number($("videoOffset").value || 0);
    for (const el of [$("operatorVideo"), $("screenVideo")]) {
      if (el.src && !el.hidden && Math.abs(el.currentTime - t) > 0.25 && t >= 0) el.currentTime = t;
    }
  }

  function stopPlay() {
    state.playing = false;
    clearInterval(state.playTimer);
    $("play").textContent = "▶ Play";
    for (const el of [$("operatorVideo"), $("screenVideo")]) if (el.src) el.pause();
  }
  $("play").onclick = () => {
    if (state.playing) return stopPlay();
    if (state.buffer.length < 2) return;
    state.playing = true;
    $("play").textContent = "❚❚ Pause";
    let wall = performance.now(), simMs = current() ? current().timestamp_utc_ms : state.buffer[0].timestamp_utc_ms;
    for (const el of [$("operatorVideo"), $("screenVideo")]) if (el.src && !el.hidden) el.play().catch(() => {});
    state.playTimer = setInterval(() => {
      const now = performance.now();
      simMs += now - wall; wall = now;
      let i = state.cursor;
      while (i < state.buffer.length - 1 && state.buffer[i + 1].timestamp_utc_ms <= simMs) i++;
      state.cursor = i; state.dirtyTimeline = true;
      if (i >= state.buffer.length - 1) stopPlay();
    }, 33);
  };

  $("fileAoi").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const data = JSON.parse(await f.text());
    aois = Array.isArray(data) ? data : data.aois;
  });

  $("exportJsonl").onclick = () => {
    const blob = new Blob([state.buffer.map((p) => JSON.stringify(p)).join("\n") + "\n"], { type: "application/x-ndjson" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `cogsense-session-${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl`;
    a.click();
  };

  // ---------------------------------------------------------------- render loop
  let lastTl = 0;
  function frame(ts) {
    renderHud();
    if (state.dirtyTimeline && ts - lastTl > 200) { drawTimeline(); state.dirtyTimeline = false; lastTl = ts; }
    requestAnimationFrame(frame);
  }
  window.addEventListener("resize", () => { state.dirtyTimeline = true; });
  requestAnimationFrame(frame);

  const params = new URLSearchParams(location.search);
  if (params.get("ws")) $("wsUrl").value = params.get("ws");
  if (params.get("autoconnect") !== "0") connect();
})();
