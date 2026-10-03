// Runs dashboard/engine.js over frames from a JSON file; writes payloads as JSON.
const fs = require("fs");
const path = require("path");
const { CogSenseEngine } = require(path.join(__dirname, "..", "dashboard", "engine.js"));

const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const eng = new CogSenseEngine();
const out = [];
input.frames.forEach((f, i) => {
  if (i === input.calibrate_at) eng.startCalibration(f.t);
  out.push(eng.process({ timestamp_s: f.t, landmarks: f.lm, width: f.w, height: f.h, frame_id: f.id,
    timestamp_utc_ms: f.utc, blendshapes: f.bs || undefined }));
});
fs.writeFileSync(process.argv[3], JSON.stringify(out));
