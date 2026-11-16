#!/usr/bin/env python3
"""
DiffuserCam PSF depth-sweep capture.

Moves the point light source to a series of distances from the sensor using the
Arduino stepper controller (see ../../controller), and at every distance captures
a calibration point-spread function (PSF) from the OV9281 mono sensor.

What makes the data usable for DiffuserCam reconstruction:
  * Raw, linear 10-bit sensor values (R10 raw stream, no ISP, no gamma, no
    auto exposure / auto gain). Saved as float32 .npy, averaged over N frames.
  * Each PSF is checked for saturation (clipped PSFs ruin deconvolution) and for
    enough signal. Exposure is auto-bracketed per distance (or fixed) and the
    actual exposure from the sensor metadata is recorded so PSFs can be
    normalised against each other.
  * Only frames whose exposure started after the stage stopped and settled, and
    after the requested exposure took effect, are used.
  * Distances default to uniform spacing in 1/z. The PSF of a lensless camera
    scales with magnification (1 + t/z), so equal 1/z steps give equal changes
    in the PSF; linear spacing over-samples far depths.
  * Travel is limited in software to 320 mm from zero (38..358 mm source to
    sensor). There is no back limit switch on the rig, so the script never relies
    on one; if one is added later and reports "Back Limit", the run stops safely.
  * Stage moves are snapped to whole motor steps (800 steps/mm by default) and
    always go outward from the zero position, so lead-screw backlash stays on
    the same side.
  * Optional dark/background frames at every exposure used (light source off).
  * manifest.json / manifest.csv record distances, exposures, statistics and
    warnings. The PSF size is fitted against 1/z to check the stage really moved
    the distances requested.

Controller protocol (firmware in controller/src/main.cpp, 115200 baud):
  "ZERO\\n"       -> "ZERO_SUCCESS"   (homes to the front limit = zero_distance mm)
  "MOVE <mm>\\n"  -> "MOVE_SUCCESS" | "MOVE_FAIL - ..."   (relative move;
                    the firmware moves STEPPER_FORWARD*STEPS_PER_MM*mm steps, and
                    FORWARD points toward the front limit, so moving the light
                    AWAY from the sensor needs a negative mm value.)

Debug web UI / API (Flask, default http://<pi>:8002/):
  GET  /                    live view + status page
  GET  /stream.mjpg         live MJPEG (?stretch=0|1 &scale=0.5)
  GET  /api/status          JSON run state, current distance, last PSF stats
  GET  /api/latest.npy      latest raw frame (uint16, 10-bit values)
  GET  /api/last_capture.jpg   last saved PSF (stretched)
  POST /api/pause | /api/resume | /api/abort
  POST /api/zero, /api/goto?distance_mm=X   (only in --preview mode)

Examples:
  # 20 PSFs from 38 mm to 300 mm (max is 358 mm = 38 + 320 mm travel), uniform in 1/z, auto exposure, dark frames at end
  python3 psf_capture.py --stop 300 --num 20

  # explicit distances, fixed exposure, no dark frames
  python3 psf_capture.py --distances 38,50,75,100,150 --exposure-us 8000 --no-dark

  # camera + web viewer only (for aligning the point source)
  python3 psf_capture.py --preview

  # dry run without the Arduino attached
  python3 psf_capture.py --port sim --stop 120 --num 5 --no-dark --yes
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("psf_capture")

SENSOR_SIZE = (1280, 800)      # OV9281 full resolution (w, h)
SENSOR_MAX = 1023              # 10-bit full scale
MAX_TRAVEL_MM = 320.0          # physical travel away from the ZERO position
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Controller (Arduino over USB serial)
# --------------------------------------------------------------------------- #
class ControllerError(RuntimeError):
    pass


class LimitHit(ControllerError):
    def __init__(self, which: str, msg: str):
        super().__init__(msg)
        self.which = which  # "front" | "back"


class StageController:
    """Tracks the light-source position in whole motor steps from zero."""

    def __init__(self, port: str, baud: int, steps_per_mm: float, away_sign: int,
                 zero_timeout: float, move_speed_mm_s: float, max_travel_mm: float):
        self.port = port
        self.baud = baud
        self.steps_per_mm = steps_per_mm
        self.away_sign = away_sign
        self.zero_timeout = zero_timeout
        self.move_speed = move_speed_mm_s
        # Software travel limit. This is the protection against running off the
        # back of the rail; the back limit switch may not exist, so it's never relied on.
        self.max_steps = int(math.floor(max_travel_mm * steps_per_mm + 1e-6))
        self.position_steps: int | None = None   # None = unknown, needs ZERO
        self.lock = threading.Lock()
        self._ser = None

    # -- connection --
    def open(self):
        import serial  # pyserial
        log.info("Opening controller on %s @ %d", self.port, self.baud)
        self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        # Opening the port toggles DTR, which resets the Nano. Wait for the
        # bootloader to finish, then drop anything it printed.
        time.sleep(2.5)
        self._ser.reset_input_buffer()

    def close(self):
        if self._ser:
            self._ser.close()
            self._ser = None

    # -- low-level --
    def _command(self, cmd: str, ok: str, timeout: float) -> str:
        assert self._ser is not None
        self._ser.reset_input_buffer()
        log.debug("-> %s", cmd)
        self._ser.write((cmd + "\n").encode("ascii"))
        self._ser.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self._ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            log.debug("<- %s", line)
            if line.startswith(ok):
                return line
            if "FAIL" in line or line.startswith("ERROR"):
                if "Front Limit" in line:
                    raise LimitHit("front", line)
                if "Back Limit" in line:   # only if a back switch is ever fitted
                    raise LimitHit("back", line)
                raise ControllerError(line)
            log.info("controller: %s", line)
        raise ControllerError(f"Timeout after {timeout:.0f}s waiting for '{ok}' to '{cmd}'")

    # -- high-level --
    def zero(self):
        with self.lock:
            self.position_steps = None
            self._command("ZERO", "ZERO_SUCCESS", self.zero_timeout)
            self.position_steps = 0

    def mm_to_steps(self, travel_mm: float) -> int:
        return int(round(travel_mm * self.steps_per_mm))

    def move_to_steps(self, target_steps: int):
        """Absolute move (in steps away from zero) with exact step bookkeeping."""
        with self.lock:
            if self.position_steps is None:
                raise ControllerError("Position unknown: run ZERO first")
            if not 0 <= target_steps <= self.max_steps:
                raise ControllerError(
                    f"Target {target_steps / self.steps_per_mm:.3f} mm travel is outside the "
                    f"allowed range 0..{self.max_steps / self.steps_per_mm:.3f} mm; move refused")
            delta = target_steps - self.position_steps
            if delta == 0:
                return
            fw_steps = self.away_sign * delta          # steps in firmware's frame
            # Firmware does long(800.0f * atof(arg)) which truncates toward zero.
            # Add half a step so float rounding still lands on the exact count.
            arg = (fw_steps + math.copysign(0.5, fw_steps)) / self.steps_per_mm
            dist_mm = abs(delta) / self.steps_per_mm
            timeout = 15.0 + dist_mm / self.move_speed
            try:
                self._command(f"MOVE {arg:.6f}", "MOVE_SUCCESS", timeout)
            except ControllerError:
                self.position_steps = None   # stage stopped somewhere unknown
                raise
            self.position_steps = target_steps

    @property
    def travel_mm(self) -> float | None:
        if self.position_steps is None:
            return None
        return self.position_steps / self.steps_per_mm


class SimulatedSerial:
    """Stands in for the Arduino so the pipeline can be tested without hardware."""

    def __init__(self, steps_per_mm: float, travel_mm: float = MAX_TRAVEL_MM,
                 back_limit_switch: bool = False):
        self.spm = steps_per_mm
        self.pos = None
        self.travel = int(travel_mm * steps_per_mm)
        self.back_switch = back_limit_switch
        self._out: list[bytes] = []

    def reset_input_buffer(self):
        self._out.clear()

    def flush(self):
        pass

    def close(self):
        pass

    def write(self, data: bytes):
        parts = data.decode().strip().split()
        cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
        if cmd == "ZERO":
            time.sleep(1.0)
            self.pos = 0
            self._out.append(b"ZERO_SUCCESS\r\n")
        elif cmd == "MOVE" and arg is not None:
            steps = int(np.float32(self.spm) * np.float32(float(arg)))  # emulate AVR
            time.sleep(min(abs(steps) / self.spm / 5.0, 5.0))
            new = self.pos - steps  # firmware FORWARD = toward front limit
            if new < 0:
                self.pos = 0
                self._out.append(b"MOVE_FAIL - Front Limit Triggered\r\n")
            elif new > self.travel:
                self.pos = self.travel
                if self.back_switch:
                    self._out.append(b"MOVE_FAIL - Back Limit Triggered\r\n")
                else:   # no switch: carriage stalls at the hard stop, firmware still says OK
                    log.error("SIM: carriage driven into the back hard stop!")
                    self._out.append(b"MOVE_SUCCESS\r\n")
            else:
                self.pos = new
                self._out.append(b"MOVE_SUCCESS\r\n")
        else:
            self._out.append(b"ERROR - Not a valid command\r\n")

    def readline(self) -> bytes:
        if self._out:
            return self._out.pop(0)
        time.sleep(0.05)
        return b""


def find_serial_port() -> str:
    candidates = sorted(glob.glob("/dev/serial/by-id/*")) + \
        sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    if not candidates:
        raise ControllerError("No USB serial device found (is the Arduino plugged in?). "
                              "Use --port /dev/ttyXXX or --port sim.")
    return candidates[0]


# --------------------------------------------------------------------------- #
# Camera (Picamera2, raw 10-bit)
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    seq: int
    raw: np.ndarray          # uint16 (h, w), 0..1023
    timestamp_ns: int        # SensorTimestamp, CLOCK_MONOTONIC
    exposure_us: int
    gain: float


class Camera:
    """Owns the Picamera2 instance. A background thread keeps grabbing raw frames;
    the web stream reads the latest one and the sweep waits for qualifying ones."""

    def __init__(self, gain: float, exposure_us: int):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (640, 400), "format": "YUV420"},   # required, unused
            raw={"format": "R10", "size": SENSOR_SIZE},
            buffer_count=4,
            controls=self._exposure_controls(exposure_us) | {
                "AeEnable": False,
                "AnalogueGain": float(gain),
                "NoiseReductionMode": 0,
            },
        )
        self.picam2.configure(cfg)
        self.raw_cfg = self.picam2.camera_configuration()["raw"]
        self.requested_exposure = exposure_us
        self.gain = gain
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @staticmethod
    def _exposure_controls(exposure_us: int) -> dict:
        frame_us = max(33333, int(exposure_us) + 2000)
        return {"ExposureTime": int(exposure_us), "FrameDurationLimits": (frame_us, frame_us)}

    def start(self):
        self.picam2.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.picam2.stop()
        self.picam2.close()

    def _loop(self):
        w, h = self.raw_cfg["size"]
        while self._running:
            try:
                req = self.picam2.capture_request()
                try:
                    buf = req.make_array("raw")
                    md = req.get_metadata()
                finally:
                    req.release()
            except Exception as e:  # keep the thread alive, surface the error
                self.error = repr(e)
                log.exception("camera capture failed")
                time.sleep(0.5)
                continue
            # R10 is unpacked: little-endian 16-bit words, values 0..1023.
            raw = buf.view("<u2")[:h, :w]
            with self._cond:
                self._seq += 1
                self._latest = Frame(self._seq, raw, int(md.get("SensorTimestamp", time.monotonic_ns())),
                                     int(md.get("ExposureTime", 0)), float(md.get("AnalogueGain", 0.0)))
                self._cond.notify_all()

    def latest(self) -> Frame | None:
        with self._cond:
            return self._latest

    def wait_newer(self, seq: int, timeout: float = 2.0) -> Frame | None:
        with self._cond:
            self._cond.wait_for(lambda: self._latest is not None and self._latest.seq > seq, timeout)
            return self._latest

    def set_exposure(self, exposure_us: int):
        self.requested_exposure = int(exposure_us)
        self.picam2.set_controls(self._exposure_controls(exposure_us))

    @staticmethod
    def exposure_matches(actual: int, requested: int) -> bool:
        # The sensor quantises exposure to whole line times (~9 us).
        return abs(actual - requested) <= max(0.02 * requested, 20)

    def grab(self, n: int, not_before_ns: int, timeout: float | None = None) -> list[Frame]:
        """Return n consecutive new frames that started after `not_before_ns` and
        were taken with the currently requested exposure."""
        req = self.requested_exposure
        frame_s = max(0.0333, req / 1e6)
        timeout = timeout or (5.0 + (n + 12) * frame_s * 1.5)
        deadline = time.monotonic() + timeout
        out: list[Frame] = []
        seq = self.latest().seq if self.latest() else 0
        while len(out) < n:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Only got {len(out)}/{n} valid frames (exposure {req} us)")
            f = self.wait_newer(seq, timeout=max(1.0, 3 * frame_s))
            if f is None or f.seq == seq:
                continue
            seq = f.seq
            if f.timestamp_ns > not_before_ns and self.exposure_matches(f.exposure_us, req):
                out.append(f)
        return out


# --------------------------------------------------------------------------- #
# Image analysis
# --------------------------------------------------------------------------- #
def psf_stats(img: np.ndarray, edge_px: int = 8) -> dict:
    """Statistics used to judge a PSF. img: float array in raw DN."""
    small = img[::4, ::4]
    bg = float(np.percentile(small, 1))
    low = small[small <= np.percentile(small, 50)]
    noise = float(1.4826 * np.median(np.abs(low - np.median(low)))) + 1e-6
    peak = float(np.percentile(img, 99.995))
    w = np.clip(img - bg - 3 * noise, 0, None)
    total = float(w.sum())
    h, wd = img.shape
    stats = {"background": bg, "noise": noise, "peak": peak, "max": float(img.max()),
             "mean": float(img.mean()), "signal_sum": total,
             "snr_peak": (peak - bg) / noise}
    if total > 0:
        ys, xs = np.mgrid[0:h, 0:wd]
        cx = float((w * xs).sum() / total)
        cy = float((w * ys).sum() / total)
        r2 = float((w * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / total)
        edge = w.copy()
        edge[edge_px:-edge_px, edge_px:-edge_px] = 0
        stats |= {"centroid_x": cx, "centroid_y": cy, "rms_radius_px": math.sqrt(r2),
                  "edge_energy_frac": float(edge.sum() / total)}
    else:
        stats |= {"centroid_x": None, "centroid_y": None, "rms_radius_px": None,
                  "edge_energy_frac": None}
    return stats


def to_display(raw: np.ndarray, stretch: bool, scale: float = 1.0,
               sat_level: int = SENSOR_MAX) -> np.ndarray:
    """10-bit raw -> 8-bit BGR. Saturated pixels drawn red."""
    img = raw.astype(np.float32)
    if stretch:
        lo = float(np.percentile(img[::4, ::4], 1))
        hi = max(float(np.percentile(img[::4, ::4], 99.9)), lo + 8)
    else:
        lo, hi = 0.0, float(SENSOR_MAX)
    g = np.clip((img - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    sat = raw >= sat_level
    if sat.any():
        bgr[sat] = (0, 0, 255)
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr


# --------------------------------------------------------------------------- #
# Distance plan
# --------------------------------------------------------------------------- #
def plan_distances(args) -> list[float]:
    if args.distances:
        d = [float(x) for x in args.distances.split(",") if x.strip()]
    else:
        if args.stop is None:
            raise SystemExit("Specify --stop (and optionally --start/--num/--spacing) or --distances")
        a, b, n = args.start, args.stop, args.num
        if n < 2:
            d = [a]
        elif args.spacing == "inverse":
            d = list(1.0 / np.linspace(1.0 / a, 1.0 / b, n))
        else:
            d = list(np.linspace(a, b, n))
    # snap to motor steps, sort outward (consistent backlash), dedupe
    spm = args.steps_per_mm
    snapped = sorted({round((x - args.zero_distance) * spm) for x in d})
    out = [args.zero_distance + s / spm for s in snapped]
    for x in out:
        if x < args.zero_distance - 1e-9:
            raise SystemExit(f"Distance {x:.3f} mm is closer than the zero position "
                             f"({args.zero_distance} mm)")
        if x - args.zero_distance > args.max_travel + 1e-9:
            raise SystemExit(f"Distance {x:.3f} mm is beyond the maximum travel: the farthest "
                             f"allowed is {args.zero_distance + args.max_travel:.3f} mm "
                             f"(zero {args.zero_distance} + {args.max_travel} mm travel)")
    return out


# --------------------------------------------------------------------------- #
# Shared run state (read by the web UI)
# --------------------------------------------------------------------------- #
@dataclass
class RunState:
    phase: str = "starting"
    message: str = ""
    index: int = 0
    total: int = 0
    target_distance_mm: float | None = None
    distance_mm: float | None = None
    exposure_us: int | None = None
    run_dir: str | None = None
    last_capture: dict | None = None
    warnings: list = field(default_factory=list)
    paused: bool = False
    started: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    last_jpeg: bytes | None = field(default=None, repr=False)
    pause_evt: threading.Event = field(default_factory=threading.Event, repr=False)
    abort_evt: threading.Event = field(default_factory=threading.Event, repr=False)

    def __post_init__(self):
        self.pause_evt.set()  # set = running

    def public(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)
             if f.name not in ("last_jpeg", "pause_evt", "abort_evt")}
        d["warnings"] = list(self.warnings)
        d["aborted"] = self.abort_evt.is_set()
        return d


class Aborted(Exception):
    pass


# --------------------------------------------------------------------------- #
# Flask debug server
# --------------------------------------------------------------------------- #
INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSF Capture</title>
<style>
 body{font-family:system-ui,sans-serif;background:#111;color:#ddd;margin:0;padding:12px}
 .row{display:flex;flex-wrap:wrap;gap:12px} .card{background:#1c1c1c;padding:10px;border-radius:6px}
 img{max-width:100%;display:block;background:#000} pre{font-size:12px;white-space:pre-wrap;margin:0}
 button{margin:2px;padding:6px 12px} .warn{color:#f90}
</style></head><body>
<h3 style="margin:4px 0 10px">DiffuserCam PSF capture</h3>
<div class="row">
 <div class="card" style="flex:2 1 480px"><b>Live</b>
  <label><input type="checkbox" id="st" checked onchange="setStream()"> stretch</label>
  <span style="font-size:12px">(red = saturated)</span>
  <img id="live"></div>
 <div class="card" style="flex:1 1 320px"><b>Last saved PSF</b><img id="last">
  <div style="margin-top:8px">
   <button onclick="post('pause')">Pause</button><button onclick="post('resume')">Resume</button>
   <button onclick="if(confirm('Abort run?'))post('abort')">Abort</button></div>
  <div id="pv" style="display:none;margin-top:6px">
   <button onclick="post('zero')">Zero</button>
   <input id="gd" size="6" placeholder="mm"><button onclick="post('goto?distance_mm='+gd.value)">Go to</button></div>
  <pre id="status"></pre></div>
</div>
<script>
function setStream(){live.src='stream.mjpg?scale=0.75&stretch='+(st.checked?1:0)+'&t='+Date.now()}
function post(p){fetch('api/'+p,{method:'POST'}).then(r=>r.json()).then(j=>console.log(j))}
let lastIdx=-1;
async function poll(){try{const s=await (await fetch('api/status')).json();
 status.textContent=JSON.stringify(s,null,1);
 pv.style.display=s.phase==='preview'?'block':'none';
 const k=s.last_capture?s.last_capture.index:-1;
 if(k!==lastIdx){lastIdx=k;last.src='api/last_capture.jpg?i='+k}}catch(e){}
 setTimeout(poll,1000)}
setStream();poll();
</script></body></html>"""


def start_web(camera: Camera, state: RunState, host: str, port: int,
              stage: StageController | None, args) -> None:
    from flask import Flask, Response, jsonify, request
    from werkzeug.serving import make_server

    app = Flask("psf_capture")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.get("/stream.mjpg")
    def stream():
        stretch = request.args.get("stretch", "1") != "0"
        scale = min(max(float(request.args.get("scale", 0.5)), 0.1), 1.0)

        def gen():
            seq = 0
            while True:
                f = camera.wait_newer(seq, timeout=2.0)
                if f is None or f.seq == seq:
                    continue
                seq = f.seq
                img = to_display(f.raw, stretch, scale)
                txt = (f"{state.phase} {state.index}/{state.total}  "
                       f"z={state.distance_mm if state.distance_mm is None else round(state.distance_mm, 3)}mm  "
                       f"exp={f.exposure_us}us  max={int(f.raw.max())}")
                cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                time.sleep(0.05)  # cap ~20 fps per client

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def status():
        d = state.public()
        f = camera.latest()
        d["camera"] = {"seq": f.seq if f else None, "exposure_us": f.exposure_us if f else None,
                       "gain": f.gain if f else None, "frame_max": int(f.raw.max()) if f else None,
                       "error": camera.error}
        if stage:
            d["stage_travel_mm"] = stage.travel_mm
        return jsonify(d)

    @app.get("/api/latest.npy")
    def latest_npy():
        import io
        f = camera.latest()
        if f is None:
            return jsonify(error="no frame yet"), 503
        bio = io.BytesIO()
        np.save(bio, f.raw.copy())
        return Response(bio.getvalue(), mimetype="application/octet-stream",
                        headers={"Content-Disposition": f"attachment; filename=frame_{f.seq}.npy"})

    @app.get("/api/last_capture.jpg")
    def last_capture():
        if state.last_jpeg is None:
            return Response(status=204)
        return Response(state.last_jpeg, mimetype="image/jpeg")

    @app.post("/api/pause")
    def pause():
        state.pause_evt.clear()
        state.paused = True
        return jsonify(ok=True, paused=True)

    @app.post("/api/resume")
    def resume():
        state.pause_evt.set()
        state.paused = False
        return jsonify(ok=True, paused=False)

    @app.post("/api/abort")
    def abort():
        state.abort_evt.set()
        state.pause_evt.set()
        return jsonify(ok=True, aborted=True)

    def preview_stage_call(fn):
        if state.phase != "preview" or stage is None:
            return jsonify(error="stage control only available in --preview with a controller"), 409
        if not stage.lock.acquire(blocking=False):
            return jsonify(error="stage busy"), 409
        stage.lock.release()

        def run():
            try:
                fn()
                state.distance_mm = args.zero_distance + stage.travel_mm
                state.message = "ok"
            except Exception as e:
                state.message = f"stage error: {e}"
        threading.Thread(target=run, daemon=True).start()
        return jsonify(ok=True)

    @app.post("/api/zero")
    def api_zero():
        return preview_stage_call(stage.zero if stage else None)

    @app.post("/api/goto")
    def api_goto():
        z = float(request.args.get("distance_mm", "nan"))
        if not (args.zero_distance <= z <= args.zero_distance + args.max_travel):
            return jsonify(error="distance out of range"), 400
        return preview_stage_call(lambda: stage.move_to_steps(stage.mm_to_steps(z - args.zero_distance)))

    srv = make_server(host, port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, name="web", daemon=True).start()
    log.info("Debug web UI on http://%s:%d/", host if host != "0.0.0.0" else _local_ip(), port)


def _local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"


# --------------------------------------------------------------------------- #
# Capture logic
# --------------------------------------------------------------------------- #
class PsfSweep:
    def __init__(self, args, camera: Camera, stage: StageController, state: RunState, run_dir: Path):
        self.a = args
        self.cam = camera
        self.stage = stage
        self.state = state
        self.dir = run_dir
        self.records: list[dict] = []
        self.darks: list[dict] = []
        self.status = "running"
        self.meta = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "sensor": "OV9281 mono",
            "raw_format": camera.raw_cfg["format"],
            "raw_size_wh": list(camera.raw_cfg["size"]),
            "bit_depth": 10,
            "data_units": "raw DN (0..1023), NOT dark-subtracted, mean of n_frames",
            "zero_distance_mm": args.zero_distance,
            "steps_per_mm": args.steps_per_mm,
            "spacing": "explicit" if args.distances else args.spacing,
            "analogue_gain": args.gain,
            "frames_per_psf": args.frames,
            "settle_s": args.settle,
            "args": {k: v for k, v in vars(args).items()},
        }

    # -- helpers --
    def check_control(self):
        if self.state.abort_evt.is_set():
            raise Aborted()
        if not self.state.pause_evt.is_set():
            self.state.phase, prev = "paused", self.state.phase
            log.info("Paused (resume from web UI)")
            while not self.state.pause_evt.wait(0.5):
                pass
            if self.state.abort_evt.is_set():
                raise Aborted()
            self.state.phase = prev

    def single(self, not_before_ns: int) -> Frame:
        return self.cam.grab(1, not_before_ns)[0]

    def auto_expose(self, start_exp: int, not_before_ns: int) -> int:
        """Find the exposure that puts the PSF's brightest pixels at ~target_peak of
        full scale without saturating. Bracketing search: never returns to an exposure
        already seen saturating, and always leaves the camera set to the result."""
        a = self.a
        lo_frac, hi_frac = a.target_peak - 0.15, min(a.target_peak + 0.15, 0.95)
        exp = int(np.clip(start_exp, a.min_exposure_us, a.max_exposure_us))
        ok_exp, sat_exp = None, None      # highest unsaturated / lowest saturated seen
        for it in range(12):
            self.check_control()
            self.cam.set_exposure(exp)
            self.state.exposure_us = exp
            f = self.single(not_before_ns)
            img = f.raw.astype(np.float32)
            bg = float(np.percentile(img[::4, ::4], 1))
            peak = float(np.percentile(img, 99.999))   # ~10 brightest px, ignores lone hot px
            nsat = int((f.raw >= a.sat_level).sum())
            frac = (peak - bg) / (SENSOR_MAX - bg)
            log.debug("  AE it%d exp=%dus peak=%.0f bg=%.0f frac=%.2f sat=%d", it, exp, peak, bg, frac, nsat)
            if nsat > a.max_saturated:
                sat_exp = exp if sat_exp is None else min(sat_exp, exp)
                new = exp * 0.4 if ok_exp is None else math.sqrt(ok_exp * sat_exp)
            else:
                ok_exp = exp if ok_exp is None else max(ok_exp, exp)
                if lo_frac <= frac <= hi_frac:
                    break
                new = exp * a.target_peak / max(frac, 1e-3)
                if sat_exp is not None and new >= sat_exp * 0.97:
                    new = math.sqrt(exp * sat_exp)
            new = int(np.clip(new, a.min_exposure_us, a.max_exposure_us))
            if sat_exp is not None and ok_exp is not None and sat_exp <= ok_exp * 1.05:
                break       # bracket closed: ok_exp is as bright as we can go
            if new == exp:
                break       # pinned at an exposure limit; flagged later
            exp = new
        best = ok_exp if ok_exp is not None else exp
        self.cam.set_exposure(best)
        self.state.exposure_us = best
        return best

    def capture_at(self, z: float, exp: int, not_before_ns: int) -> tuple[np.ndarray, dict, int]:
        a = self.a
        for attempt in range(3):
            if a.exposure_us is None:
                exp = self.auto_expose(exp, not_before_ns)
            else:
                self.cam.set_exposure(exp)
            self.state.exposure_us = exp
            self.check_control()
            frames = self.cam.grab(a.frames, not_before_ns)
            acc = np.zeros(frames[0].raw.shape, np.float64)
            sat_max = 0
            for f in frames:
                acc += f.raw
                sat_max = max(sat_max, int((f.raw >= a.sat_level).sum()))
            mean = (acc / len(frames)).astype(np.float32)
            if sat_max > a.max_saturated and a.exposure_us is None and attempt < 2:
                log.warning("  %d saturated px during averaging, reducing exposure", sat_max)
                exp = int(exp * 0.6)
                continue
            break
        info = {
            "exposure_us": frames[0].exposure_us,
            "requested_exposure_us": exp,
            "analogue_gain": frames[0].gain,
            "n_frames": len(frames),
            "frame_seq": [f.seq for f in frames],
            "frame_timestamps_ns": [f.timestamp_ns for f in frames],
            "saturated_px_max": sat_max,
        }
        return mean, info, exp

    def save_psf(self, i: int, z: float, mean: np.ndarray, info: dict) -> dict:
        a = self.a
        stem = f"psf_{i:03d}_z{z:08.3f}mm"
        np.save(self.dir / f"{stem}.npy", mean)
        cv2.imwrite(str(self.dir / f"{stem}.png"),
                    (np.clip(mean, 0, SENSOR_MAX) * 64).astype(np.uint16))  # 16-bit preview
        st = psf_stats(mean)
        warns = []
        if info["saturated_px_max"] > a.max_saturated:
            warns.append(f"saturated ({info['saturated_px_max']} px)")
        if st["snr_peak"] < a.min_snr:
            warns.append(f"low SNR ({st['snr_peak']:.0f})")
        if (st["peak"] - st["background"]) / (SENSOR_MAX - st["background"]) < 0.25:
            warns.append("peak < 25% of full scale")
        if st["edge_energy_frac"] is not None and st["edge_energy_frac"] > 0.02:
            warns.append(f"PSF energy at sensor edge ({100 * st['edge_energy_frac']:.1f}%) - may be cropped")
        if a.exposure_us is None and info["requested_exposure_us"] >= a.max_exposure_us:
            warns.append("at max exposure")
        rec = {
            "index": i,
            "distance_mm": round(z, 4),
            "travel_mm": round(z - a.zero_distance, 6),
            "travel_steps": self.stage.position_steps,
            "file": f"{stem}.npy",
            "preview": f"{stem}.png",
            "captured": dt.datetime.now().isoformat(timespec="seconds"),
            **{k: v for k, v in info.items() if k not in ("frame_seq", "frame_timestamps_ns")},
            "exposure_normalised_signal": st["signal_sum"] / max(info["exposure_us"], 1),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()},
            "warnings": warns,
        }
        for w in warns:
            log.warning("  z=%.3f mm: %s", z, w)
            self.state.warnings.append(f"z={z:.3f}: {w}")
        ok, jpg = cv2.imencode(".jpg", to_display(mean.astype(np.uint16), True, 0.5))
        self.state.last_jpeg = jpg.tobytes() if ok else None
        self.state.last_capture = rec
        return rec

    def write_manifest(self):
        m = dict(self.meta, status=self.status, psfs=self.records, darks=self.darks,
                 validation=self.validate(), warnings=self.state.warnings)
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(m, indent=2, default=str))
        tmp.replace(self.dir / "manifest.json")
        if self.records:
            cols = ["index", "distance_mm", "travel_mm", "travel_steps", "file", "exposure_us",
                    "analogue_gain", "n_frames", "saturated_px_max", "background", "peak",
                    "snr_peak", "rms_radius_px", "centroid_x", "centroid_y", "edge_energy_frac",
                    "exposure_normalised_signal", "warnings"]
            with open(self.dir / "manifest.csv", "w", newline="") as fh:
                wr = csv.DictWriter(fh, cols, extrasaction="ignore")
                wr.writeheader()
                for r in self.records:
                    wr.writerow(r | {"warnings": "; ".join(r["warnings"])})

    def validate(self) -> dict:
        """Check that the PSF scale follows the lensless-camera model r = a(1 + t/z).
        A good fit (R^2 near 1, positive t) indicates the stage moved as commanded."""
        pts = [(r["distance_mm"], r["rms_radius_px"]) for r in self.records
               if r.get("rms_radius_px") and not r["warnings"]]
        out: dict = {"n_used": len(pts)}
        if len(pts) < 3:
            return out
        z = np.array([p[0] for p in pts])
        r = np.array([p[1] for p in pts])
        A = np.vstack([np.ones_like(z), 1 / z]).T
        (a0, a1), *_ = np.linalg.lstsq(A, r, rcond=None)
        pred = A @ np.array([a0, a1])
        ss_res = float(((r - pred) ** 2).sum())
        ss_tot = float(((r - r.mean()) ** 2).sum()) or 1e-12
        nonmono = int(np.sum(np.diff(r) > 0.02 * r[:-1]))
        out |= {"model": "rms_radius = a*(1 + t/z)", "a_px": float(a0),
                "t_mm_effective": float(a1 / a0) if a0 else None,
                "r2": 1 - ss_res / ss_tot, "radius_increases_with_z_count": nonmono}
        return out

    # -- main sweep --
    def run(self, distances: list[float]):
        a = self.a
        st = self.state
        st.total = len(distances)
        st.phase = "zeroing"
        log.info("Zeroing stage...")
        self.stage.zero()
        st.distance_mm = a.zero_distance
        exp = a.exposure_us or a.initial_exposure_us
        prev_z = None
        try:
            for i, z in enumerate(distances):
                self.check_control()
                st.index, st.target_distance_mm = i + 1, z
                target = self.stage.mm_to_steps(z - a.zero_distance)
                st.phase = "moving"
                log.info("[%d/%d] -> %.3f mm (travel %.4f mm, %d steps)",
                         i + 1, len(distances), z, z - a.zero_distance, target)
                if a.rezero_each and i > 0:
                    self.stage.zero()
                from_zero = self.stage.position_steps == 0
                try:
                    self.stage.move_to_steps(target)
                except LimitHit as e:
                    if e.which == "front" and from_zero:
                        raise ControllerError(
                            f"{e} on the first outward move: the travel direction looks inverted. "
                            f"Try --away-sign {-a.away_sign}.") from e
                    raise
                st.distance_mm = a.zero_distance + self.stage.travel_mm
                st.phase = "settling"
                time.sleep(a.settle)
                not_before = time.monotonic_ns()
                st.phase = "capturing"
                if a.exposure_us is None and prev_z is not None:
                    exp = int(exp * (z / prev_z) ** 2)   # intensity ~ 1/z^2, good AE start
                mean, info, exp = self.capture_at(z, exp, not_before)
                rec = self.save_psf(i, z, mean, info)
                self.records.append(rec)
                self.write_manifest()
                log.info("  saved %s exp=%dus peak=%.0f snr=%.0f r_rms=%s px",
                         rec["file"], rec["exposure_us"], rec["peak"], rec["snr_peak"],
                         None if rec["rms_radius_px"] is None else f"{rec['rms_radius_px']:.1f}")
                prev_z = z
            self.status = "complete"
        except LimitHit as e:
            self.status = f"stopped: {e}"
            log.error("Limit switch hit at target %.3f mm: %s", st.target_distance_mm, e)
            if e.which == "back":
                log.error("The back switch tripped inside the %.0f mm software limit: the rail "
                          "may be shorter than --max-travel, or steps were lost.", a.max_travel)
        except Aborted:
            self.status = "aborted"
            log.warning("Run aborted")
        finally:
            self.write_manifest()

    def capture_darks(self):
        a = self.a
        exps = sorted({r["requested_exposure_us"] for r in self.records})
        if not exps:
            return
        self.state.phase = "waiting for dark"
        self.state.message = "Turn the point source OFF, then press Enter in the terminal"
        print("\n>>> Turn the point light source OFF (leave everything else as-is) and press Enter "
              "to capture dark/background frames, or type 's' + Enter to skip: ", end="", flush=True)
        try:
            ans = input().strip().lower()
        except EOFError:
            ans = "s"
        if ans.startswith("s"):
            log.info("Skipping dark frames")
            return
        self.state.phase = "dark frames"
        for exp in exps:
            self.check_control()
            self.cam.set_exposure(exp)
            frames = self.cam.grab(a.dark_frames, time.monotonic_ns())
            mean = np.mean([f.raw for f in frames], axis=0, dtype=np.float64).astype(np.float32)
            name = f"dark_exp{exp:08d}us.npy"
            np.save(self.dir / name, mean)
            rec = {"file": name, "requested_exposure_us": exp, "exposure_us": frames[0].exposure_us,
                   "n_frames": len(frames), "mean": float(mean.mean()), "max": float(mean.max())}
            self.darks.append(rec)
            for r in self.records:
                if r["requested_exposure_us"] == exp:
                    r["dark_file"] = name
            log.info("  dark %s mean=%.1f", name, rec["mean"])
            if rec["max"] > 0.5 * SENSOR_MAX:
                self.state.warnings.append(f"dark at {exp}us has bright pixels "
                                           f"(max {rec['max']:.0f}) - was the source off?")
        self.write_manifest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DiffuserCam PSF depth-sweep capture",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("controller")
    g.add_argument("--port", default="auto", help="serial port, 'auto', or 'sim' for a simulated controller")
    g.add_argument("--baud", type=int, default=115200)
    g.add_argument("--zero-distance", type=float, default=38.0, help="source-to-sensor distance (mm) at ZERO")
    g.add_argument("--steps-per-mm", type=float, default=800.0, help="must match firmware STEPS_PER_MM")
    g.add_argument("--away-sign", type=int, choices=(-1, 1), default=-1,
                   help="sign of the MOVE argument that moves the source AWAY from the sensor")
    g.add_argument("--max-travel", type=float, default=MAX_TRAVEL_MM,
                   help=f"software limit on travel from zero (mm); can only be lowered below {MAX_TRAVEL_MM}")
    g.add_argument("--zero-timeout", type=float, default=None,
                   help="seconds; default allows homing from full travel at --move-speed")
    g.add_argument("--sim-back-limit", action="store_true",
                   help="with --port sim, emulate a back limit switch (not fitted on the real rig)")
    g.add_argument("--move-speed", type=float, default=0.5,
                   help="conservative speed (mm/s) used only to size move timeouts")
    g.add_argument("--rezero-each", action="store_true",
                   help="home before every position (slower; no accumulated open-loop error)")
    g.add_argument("--no-home", action="store_true", help="don't return to zero at the end")

    g = p.add_argument_group("distances")
    g.add_argument("--start", type=float, default=38.0, help="first distance (mm)")
    g.add_argument("--stop", type=float,
                   help=f"last distance (mm); at most zero-distance + max-travel ({38 + MAX_TRAVEL_MM:.0f} by default)")
    g.add_argument("--num", type=int, default=20, help="number of distances")
    g.add_argument("--spacing", choices=("inverse", "linear"), default="inverse",
                   help="inverse = uniform in 1/z (matches PSF scaling)")
    g.add_argument("--distances", help="explicit comma-separated list in mm (overrides start/stop/num)")

    g = p.add_argument_group("camera")
    g.add_argument("--frames", type=int, default=10, help="frames averaged per PSF")
    g.add_argument("--settle", type=float, default=1.0, help="seconds to wait after a move")
    g.add_argument("--exposure-us", type=int, default=None, help="fixed exposure (disables auto exposure)")
    g.add_argument("--initial-exposure-us", type=int, default=5000)
    g.add_argument("--min-exposure-us", type=int, default=20)
    g.add_argument("--max-exposure-us", type=int, default=500000)
    g.add_argument("--gain", type=float, default=1.0, help="analogue gain (1.0 = lowest noise)")
    g.add_argument("--target-peak", type=float, default=0.75, help="AE target peak as fraction of full scale")
    g.add_argument("--sat-level", type=int, default=SENSOR_MAX, help="raw value counted as saturated")
    g.add_argument("--max-saturated", type=int, default=0, help="saturated pixels tolerated per frame")
    g.add_argument("--min-snr", type=float, default=50.0, help="warn below this peak SNR")
    g.add_argument("--dark-frames", type=int, default=30)
    g.add_argument("--no-dark", action="store_true", help="skip the dark-frame step")

    g = p.add_argument_group("output / web")
    g.add_argument("--out-dir", default=str(REPO_ROOT / "camera_node" / "data"))
    g.add_argument("--host", default="0.0.0.0")
    g.add_argument("--http-port", type=int, default=8002)
    g.add_argument("--no-web", action="store_true")
    g.add_argument("--preview", action="store_true",
                   help="camera + web UI only (stage via /api/zero, /api/goto if a controller is found)")
    g.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    g.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if not 0 < args.max_travel <= MAX_TRAVEL_MM:
        p.error(f"--max-travel must be in (0, {MAX_TRAVEL_MM}] mm")
    if args.zero_timeout is None:
        args.zero_timeout = 30.0 + args.max_travel / args.move_speed
    return args


def make_stage(args, required: bool) -> StageController | None:
    try:
        port = find_serial_port() if args.port == "auto" else args.port
    except ControllerError:
        if required:
            raise
        log.warning("No controller found; stage control disabled")
        return None
    stage = StageController(port, args.baud, args.steps_per_mm, args.away_sign,
                            args.zero_timeout, args.move_speed, args.max_travel)
    if port == "sim":
        stage._ser = SimulatedSerial(args.steps_per_mm, back_limit_switch=args.sim_back_limit)
    else:
        stage.open()
    return stage


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("picamera2", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    distances = None if args.preview else plan_distances(args)
    if distances:
        print(f"\nPlanned {len(distances)} PSF distances (mm, source to sensor; zero = {args.zero_distance}):")
        for i, z in enumerate(distances):
            print(f"  {i:3d}: {z:9.4f}   travel {z - args.zero_distance:9.4f} mm")
        if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
            return 1

    state = RunState()
    cam = Camera(args.gain, args.exposure_us or args.initial_exposure_us)
    cam.start()
    stage = None
    try:
        stage = make_stage(args, required=not args.preview)
        if not args.no_web:
            start_web(cam, state, args.host, args.http_port, stage, args)

        if args.preview:
            state.phase = "preview"
            log.info("Preview mode - Ctrl+C to exit")
            while not state.abort_evt.wait(1.0):
                pass
            return 0

        run_dir = Path(args.out_dir) / dt.datetime.now().strftime("psf_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        fh = logging.FileHandler(run_dir / "capture.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logging.getLogger().addHandler(fh)
        state.run_dir = str(run_dir)
        log.info("Saving to %s", run_dir)

        sweep = PsfSweep(args, cam, stage, state, run_dir)
        try:
            sweep.run(distances)
            if not args.no_home:
                state.phase = "homing"
                log.info("Returning to zero")
                try:
                    stage.zero()
                    state.distance_mm = args.zero_distance
                except ControllerError as e:
                    log.error("Homing failed: %s", e)
            if not args.no_dark and sweep.records and not state.abort_evt.is_set():
                sweep.capture_darks()
        except KeyboardInterrupt:
            sweep.status = "interrupted"
            sweep.write_manifest()
            raise

        v = sweep.validate()
        state.phase = "done"
        print(f"\nRun {sweep.status}: {len(sweep.records)}/{len(distances)} PSFs in {run_dir}")
        if "r2" in v:
            print(f"Scale check (rms radius vs 1/z): R^2={v['r2']:.4f}, effective t={v['t_mm_effective']:.2f} mm, "
                  f"non-monotonic steps={v['radius_increases_with_z_count']}")
            if v["r2"] < 0.95 or v["radius_increases_with_z_count"]:
                print("  ! PSF size doesn't scale cleanly with distance - check the stage moved, "
                      "alignment, and stray light.")
        if state.warnings:
            print(f"{len(state.warnings)} warning(s):")
            for w in state.warnings:
                print("  -", w)
        return 0 if sweep.status == "complete" else 2
    except KeyboardInterrupt:
        log.warning("Interrupted")
        return 130
    finally:
        cam.stop()
        if stage:
            stage.close()


if __name__ == "__main__":
    sys.exit(main())
