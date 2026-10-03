/* On-device camera pipeline: getUserMedia → MediaPipe Face Landmarker (WASM) → CogSense engine.
 *
 * Built for iPhone Safari (iOS 16.4+) and works the same in desktop browsers:
 *  - frames never leave the device and are never stored; only landmarks go to the engine
 *  - getUserMedia is called first, inside the tap that started us, so Safari shows its prompt
 *  - front camera, muted + playsinline video (iOS refuses inline playback otherwise)
 *  - GPU delegate with automatic CPU fallback (WebGL support varies across iOS versions)
 *  - handles backgrounding / interruptions and keeps the screen awake while running
 *
 * MediaPipe is loaded from jsDelivr by default. Override with ?mp=<base url> (a folder holding
 * vision_bundle.mjs and wasm/) and ?model=<url> to self-host or run on a closed network.
 */
(function () {
  "use strict";

  const MP_VERSION = "0.10.21";
  const DEFAULT_MP = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
  const DEFAULT_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

  const IN_APP_UA = /FBAN|FBAV|Instagram|Line\/|MicroMessenger|GSA\/|Snapchat|TikTok|\bwv\b/i;

  class CameraError extends Error {
    constructor(code, message) { super(message); this.code = code; }
  }

  function explain(err) {
    if (err instanceof CameraError) return err;
    const name = err && err.name;
    if (name === "NotAllowedError" || name === "SecurityError")
      return new CameraError("denied", "Camera permission was denied. On iPhone: tap the “aA” icon in the address bar → Website Settings → Camera → Allow, then reload.");
    if (name === "NotFoundError" || name === "OverconstrainedError")
      return new CameraError("no-camera", "No usable camera was found on this device.");
    if (name === "NotReadableError" || name === "AbortError")
      return new CameraError("busy", "The camera is in use by another app. Close it (FaceTime, Camera, a video call) and try again.");
    return new CameraError("error", (err && err.message) || String(err));
  }

  class CogSenseCamera {
    /**
     * @param {object} o
     * @param {HTMLVideoElement} o.video
     * @param {(payload:object, landmarks:number[][]|null)=>void} o.onPayload
     * @param {(status:string, detail?:string)=>void} [o.onStatus]
     * @param {object} [o.config]   engine config overrides
     */
    constructor(o) {
      this.video = o.video;
      this.onPayload = o.onPayload;
      this.onStatus = o.onStatus || (() => {});
      const q = new URLSearchParams(location.search);
      this.mpBase = (q.get("mp") || o.mpBase || DEFAULT_MP).replace(/\/$/, "");
      this.modelUrl = q.get("model") || o.modelUrl || DEFAULT_MODEL;
      this.engine = new window.CogSense.CogSenseEngine(o.config);
      this.stream = null;
      this.landmarker = null;
      this.delegate = null;
      this.running = false;
      this.frameId = 0;
      this.lastTs = -1;
      this.lastVideoTime = -1;
      this.wakeLock = null;
      this.loopHandle = null;
      this.luma = null;
      this.lumaCanvas = null;
      this._starting = false;
      this._onVisibility = this._onVisibility.bind(this);
    }

    static support() {
      if (!window.isSecureContext) return { ok: false, code: "insecure", message: "The camera needs a secure (https) page." };
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
        return { ok: false, code: "unsupported", message: IN_APP_UA.test(navigator.userAgent)
          ? "This in-app browser blocks the camera. Open this page in Safari." : "This browser does not support camera access." };
      if (typeof WebAssembly !== "object") return { ok: false, code: "no-wasm", message: "WebAssembly is required." };
      return { ok: true };
    }

    // -------------------------------------------------------------- start / stop
    async start() {
      if (this.running || this._starting) return;
      this._starting = true;
      if (this.stream || this.landmarker) this.stop(); // restart after an interruption: release stale resources
      try {
        const s = CogSenseCamera.support();
        if (!s.ok) throw new CameraError(s.code, s.message);
        this.onStatus("requesting-camera");
        // Request the camera first so the permission prompt stays tied to the user's tap.
        this.stream = await this._getStream();
        const v = this.video;
        v.muted = true;
        v.setAttribute("playsinline", "");
        v.setAttribute("webkit-playsinline", "");
        v.srcObject = this.stream;
        this.stream.getVideoTracks().forEach((t) => t.addEventListener("ended", () => this._interrupted("The camera stopped (app was backgrounded or another app took it).")));
        await v.play();
        await this._whenReady(v);
        this.onStatus("loading-model");
        await this._loadLandmarker();
        document.addEventListener("visibilitychange", this._onVisibility);
        await this._acquireWakeLock();
        this.running = true;
        this.engine = new window.CogSense.CogSenseEngine(this.engine.cfg); // fresh state per session
        this.frameId = 0;
        this.onStatus("running", this.delegate);
        this._schedule();
      } catch (e) {
        this.stop();
        const err = explain(e);
        this.onStatus("error", err.message);
        throw err;
      } finally {
        this._starting = false;
      }
    }

    stop() {
      this.running = false;
      if (this.loopHandle !== null) {
        if (this._useRvfc && this.video.cancelVideoFrameCallback) this.video.cancelVideoFrameCallback(this.loopHandle);
        else cancelAnimationFrame(this.loopHandle);
        this.loopHandle = null;
      }
      document.removeEventListener("visibilitychange", this._onVisibility);
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
      if (this.video) { this.video.pause(); this.video.srcObject = null; }
      if (this.landmarker) { try { this.landmarker.close(); } catch (_) { /* already closed */ } this.landmarker = null; }
      if (this.wakeLock) { try { this.wakeLock.release(); } catch (_) { /* ignore */ } this.wakeLock = null; }
    }

    calibrate() {
      this.engine.startCalibration(performance.now() / 1000);
    }

    // -------------------------------------------------------------- camera
    async _getStream() {
      const attempts = [
        { video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 } }, audio: false },
        { video: { facingMode: "user" }, audio: false },
        { video: true, audio: false },
      ];
      let last;
      for (const c of attempts) {
        try { return await navigator.mediaDevices.getUserMedia(c); }
        catch (e) {
          last = e;
          if (e.name !== "OverconstrainedError" && e.name !== "TypeError") throw e; // denied/busy won't improve with looser constraints
        }
      }
      throw last;
    }

    _whenReady(v) {
      if (v.readyState >= 2 && v.videoWidth) return Promise.resolve();
      return new Promise((res, rej) => {
        const t = setTimeout(() => rej(new CameraError("busy", "The camera did not deliver video.")), 8000);
        v.addEventListener("loadeddata", () => { clearTimeout(t); res(); }, { once: true });
      });
    }

    // -------------------------------------------------------------- model
    async _loadLandmarker() {
      const url = new URL(`${this.mpBase}/vision_bundle.mjs`, location.href).href;
      const wasm = new URL(`${this.mpBase}/wasm`, location.href).href;
      let mod, fileset;
      try {
        mod = await import(/* webpackIgnore: true */ url);
        fileset = await mod.FilesetResolver.forVisionTasks(wasm);
      } catch (e) {
        throw new CameraError("model", "Could not load the face-tracking runtime (network blocked?). " + (e && e.message ? e.message : ""));
      }
      const make = (delegate) => mod.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: this.modelUrl, delegate },
        runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true,
        minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
      });
      const warm = (lm) => lm.detectForVideo(this.video, Math.max(1, Math.round(performance.now())));
      for (const delegate of ["GPU", "CPU"]) {
        try {
          const lm = await make(delegate);
          warm(lm); // GPU init can fail on the first frame rather than at creation (seen on some iOS builds)
          this.landmarker = lm;
          this.delegate = delegate;
          this.lastTs = Math.round(performance.now());
          return;
        } catch (e) {
          if (delegate === "CPU") throw new CameraError("model", "Face tracker failed to start: " + (e && e.message ? e.message : e));
        }
      }
    }

    // -------------------------------------------------------------- loop
    _schedule() {
      if (!this.running) return;
      const v = this.video;
      this._useRvfc = typeof v.requestVideoFrameCallback === "function";
      if (this._useRvfc) this.loopHandle = v.requestVideoFrameCallback((now) => this._tick(now));
      else this.loopHandle = requestAnimationFrame((now) => this._tick(now));
    }

    _tick(now) {
      if (!this.running) return;
      try { this._processFrame(now); }
      catch (e) { this.onStatus("warn", e && e.message ? e.message : String(e)); }
      this._schedule();
    }

    _processFrame(now) {
      const v = this.video;
      if (v.readyState < 2 || !v.videoWidth || v.paused) return;
      if (!this._useRvfc) { // rAF fallback: skip if the camera hasn't produced a new frame
        if (v.currentTime === this.lastVideoTime) return;
        this.lastVideoTime = v.currentTime;
      }
      if (this.frameId % 15 === 0) this._sampleLuma();
      const ts = Math.max(Math.round(now), this.lastTs + 1); // VIDEO mode needs strictly increasing ms
      this.lastTs = ts;
      const t0 = performance.now();
      const res = this.landmarker.detectForVideo(v, ts);
      const infMs = performance.now() - t0;
      const raw = res.faceLandmarks && res.faceLandmarks[0];
      const lm = raw ? raw.map((p) => [p.x, p.y, p.z]) : null;
      let bs;
      const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
      if (cats) { bs = {}; for (const c of cats) bs[c.categoryName] = c.score; }
      const payload = this.engine.process({
        timestamp_s: now / 1000, landmarks: lm, width: v.videoWidth, height: v.videoHeight, frame_id: this.frameId++,
        timestamp_utc_ms: Math.round(performance.timeOrigin + now), blendshapes: bs, luma_mean: this.luma === null ? undefined : this.luma,
      });
      payload.inference_ms = Math.round(infMs * 100) / 100;
      this.onPayload(payload, lm);
    }

    _sampleLuma() {
      try {
        if (!this.lumaCanvas) { this.lumaCanvas = document.createElement("canvas"); this.lumaCanvas.width = 32; this.lumaCanvas.height = 24; }
        const c = this.lumaCanvas.getContext("2d", { willReadFrequently: true });
        c.drawImage(this.video, 0, 0, 32, 24);
        const d = c.getImageData(0, 0, 32, 24).data;
        let s = 0;
        for (let i = 0; i < d.length; i += 4) s += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
        this.luma = s / (d.length / 4);
      } catch (_) { this.luma = null; }
    }

    // -------------------------------------------------------------- lifecycle
    async _acquireWakeLock() {
      try { if ("wakeLock" in navigator) this.wakeLock = await navigator.wakeLock.request("screen"); }
      catch (_) { /* not available or denied; harmless */ }
    }

    _interrupted(msg) {
      if (!this.running) return;
      this.running = false;
      this.onStatus("interrupted", msg);
    }

    async _onVisibility() {
      if (document.visibilityState !== "visible" || !this.stream) return;
      const live = this.stream.getVideoTracks().some((t) => t.readyState === "live");
      if (!live) return this._interrupted("The camera stopped while the page was in the background. Tap Start camera to resume.");
      await this._acquireWakeLock(); // wake locks are released when the page is hidden
      try { await this.video.play(); } catch (_) { /* needs a gesture; the tap-to-restart UI covers it */ }
    }
  }

  window.CogSenseCamera = CogSenseCamera;
})();
