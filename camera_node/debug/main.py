#!/usr/bin/env python3
"""
OV9281 (mono) MJPEG streamer with a web page for adjusting exposure time.

Run it, then open  http://<pi-address>:8002/  in a browser.

  /               web page: live image + exposure controls
  /video_feed     raw MJPEG stream (usable in any <img> tag)
  /exposure       GET -> current exposure, POST {"exposure_us": N} -> set it
"""
import math
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
WIDTH, HEIGHT = 1280, 800        # native 1 MP mode of the OV9281
DEFAULT_EXPOSURE_US = 5000       # starting exposure time (microseconds)
EXPOSURE_MIN_US = 50             # slider / input limits (microseconds)
EXPOSURE_MAX_US = 100_000
MIN_FRAME_US = 8333              # ~120 fps, the sensor's fastest rate at 1280x800
ANALOGUE_GAIN = 1.0
STREAM_FPS = 30                  # max rate sent to the browser (keeps CPU low)
JPEG_QUALITY = 80
PORT = 8002

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
_exposure_us = DEFAULT_EXPOSURE_US


def exposure_controls(exposure_us):
    """Camera controls for a given manual exposure time.

    The sensor can't expose for longer than one frame, so the frame duration
    is lengthened along with the exposure. Without this, anything above
    ~8 ms would be silently clipped.

    No AWB / ColourGains here: the OV9281 is monochrome, so they don't apply.
    """
    frame_us = max(MIN_FRAME_US, exposure_us + 500)
    return {
        "AeEnable": False,                       # no auto exposure
        "ExposureTime": exposure_us,
        "AnalogueGain": ANALOGUE_GAIN,           # note the spelling: Analogue
        "FrameDurationLimits": (frame_us, frame_us),
    }


picam2 = Picamera2()

# The main stream must use an ISP output format. "SBGGR10" is a raw Bayer
# format (and the OV9281 is mono anyway), so it can't be used here. YUV420
# works for mono sensors and its Y plane is the grayscale image.
config = picam2.create_video_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "YUV420"},
    controls=exposure_controls(DEFAULT_EXPOSURE_US),   # applied from frame 1
)
picam2.configure(config)


def clamp_exposure(value):
    return int(max(EXPOSURE_MIN_US, min(EXPOSURE_MAX_US, value)))


def set_exposure(exposure_us):
    """Apply a new exposure time (clamped to the allowed range) and return it."""
    global _exposure_us
    exposure_us = clamp_exposure(exposure_us)
    picam2.set_controls(exposure_controls(exposure_us))
    _exposure_us = exposure_us
    return exposure_us


# ----------------------------------------------------------------------------
# Frame pipeline: ONE capture thread, any number of viewers
# ----------------------------------------------------------------------------
class LatestFrame:
    """Holds the newest JPEG and wakes up every waiting viewer when it changes."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0

    def publish(self, jpeg):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_for_new(self, last_seq, timeout=5.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            return self._seq, self._jpeg


latest = LatestFrame()


def draw_label(img, text, origin=(20, 40)):
    """Draw text with a dark outline so it's readable on bright and dark scenes."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, origin, font, 0.7, 0, 4, cv2.LINE_AA)
    cv2.putText(img, text, origin, font, 0.7, 255, 2, cv2.LINE_AA)


def capture_loop():
    """Grab frames, annotate them and encode to JPEG, once, for all viewers."""
    interval = 1.0 / STREAM_FPS
    last_sent = 0.0
    while True:
        try:
            req = picam2.capture_request()
            try:
                if time.monotonic() - last_sent < interval:
                    continue  # too soon for the stream rate; drop this frame
                yuv = req.make_array("main")        # shape (HEIGHT*3/2, WIDTH)
                meta = req.get_metadata()
                gray = cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_I420)  # Y plane only
            finally:
                req.release()
            last_sent = time.monotonic()

            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            exp = meta.get("ExposureTime", _exposure_us)
            draw_label(gray, f"OV9281 | {stamp} | exp {exp} us")

            ok, buf = cv2.imencode(".jpg", gray,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                latest.publish(buf.tobytes())
        except Exception as exc:  # keep streaming even if one frame fails
            print(f"capture error: {exc!r}")
            time.sleep(0.5)


def generate_frames():
    """Per-viewer generator: sends each new JPEG as a multipart chunk."""
    seq = 0
    while True:
        new_seq, jpeg = latest.wait_for_new(seq)
        if new_seq == seq or jpeg is None:
            continue
        seq = new_seq
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
               + jpeg + b"\r\n")


# ----------------------------------------------------------------------------
# Web page
# ----------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OV9281 camera</title>
<style>
  body { margin: 0; padding: 16px; background: #111; color: #eee;
         font: 15px system-ui, sans-serif; }
  .wrap { max-width: 1280px; margin: 0 auto; }
  img { width: 100%; height: auto; display: block; background: #000; }
  .panel { margin-top: 12px; padding: 12px 14px; background: #1c1c1c;
           border-radius: 8px; display: flex; flex-wrap: wrap;
           gap: 10px; align-items: center; }
  .panel label { font-weight: 600; }
  input[type=range] { flex: 1 1 240px; min-width: 160px; }
  input[type=number] { width: 7em; padding: 6px; background: #000; color: #eee;
                       border: 1px solid #444; border-radius: 4px; }
  button { padding: 7px 14px; border: 0; border-radius: 4px; cursor: pointer;
           background: #2f6fed; color: #fff; font-size: 15px; }
  button.alt { background: #3a3a3a; }
  button:disabled { opacity: .5; cursor: default; }
  #status { flex-basis: 100%; color: #9a9; min-height: 1.2em; }
  #status.err { color: #e88; }
</style>
</head>
<body>
<div class="wrap">
  <img src="/video_feed" alt="Live camera stream">
  <div class="panel">
    <label for="exp">Exposure time</label>
    <input id="slider" type="range" min="0" max="1000" step="1" aria-label="Exposure slider">
    <input id="exp" type="number" min="__MIN__" max="__MAX__" step="100"> <span>&micro;s</span>
    <button id="apply">Apply</button>
    <button id="half" class="alt" title="Halve the exposure (-1 stop)">&frac12;&times;</button>
    <button id="double" class="alt" title="Double the exposure (+1 stop)">2&times;</button>
    <div id="status"></div>
  </div>
</div>
<script>
const MIN = __MIN__, MAX = __MAX__;
const slider = document.getElementById('slider');
const input  = document.getElementById('exp');
const status = document.getElementById('status');
const buttons = document.querySelectorAll('button');

// The slider is logarithmic: exposure is more useful to think about in stops.
const toSlider   = us => Math.round(1000 * Math.log(us / MIN) / Math.log(MAX / MIN));
const fromSlider = v  => Math.round(MIN * Math.pow(MAX / MIN, v / 1000));

function show(us) {
  input.value = us;
  slider.value = toSlider(us);
}

function say(text, isError) {
  status.textContent = text;
  status.className = isError ? 'err' : '';
}

async function send(us) {
  us = Math.round(Number(us));
  if (!Number.isFinite(us)) { say('Enter a number of microseconds.', true); return; }
  us = Math.min(MAX, Math.max(MIN, us));
  buttons.forEach(b => b.disabled = true);
  try {
    const r = await fetch('/exposure', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({exposure_us: us})
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    show(data.exposure_us);
    say('Exposure set to ' + data.exposure_us + ' \\u00b5s (' +
        (data.exposure_us / 1000).toFixed(2) + ' ms)');
  } catch (e) {
    say('Could not set exposure: ' + e.message, true);
  } finally {
    buttons.forEach(b => b.disabled = false);
  }
}

document.getElementById('apply').onclick  = () => send(input.value);
document.getElementById('half').onclick   = () => send(Number(input.value) / 2);
document.getElementById('double').onclick = () => send(Number(input.value) * 2);
input.addEventListener('keydown', e => { if (e.key === 'Enter') send(input.value); });
slider.addEventListener('input',  () => { input.value = fromSlider(slider.value); });
slider.addEventListener('change', () => send(fromSlider(slider.value)));

// Start from whatever the camera is currently using.
fetch('/exposure').then(r => r.json()).then(d => show(d.exposure_us));
</script>
</body>
</html>
"""


@app.route("/")
def index():
    page = PAGE.replace("__MIN__", str(EXPOSURE_MIN_US)).replace("__MAX__", str(EXPOSURE_MAX_US))
    return Response(page, mimetype="text/html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream. Put this in the src attribute of an <img> tag."""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/exposure", methods=["GET", "POST"])
def exposure():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        try:
            value = float(data["exposure_us"])
            if not math.isfinite(value):       # reject NaN / infinity
                raise ValueError(value)
            set_exposure(value)
        except (KeyError, TypeError, ValueError):
            return jsonify(error="exposure_us must be a number"), 400
    return jsonify(exposure_us=_exposure_us,
                   min_us=EXPOSURE_MIN_US, max_us=EXPOSURE_MAX_US)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    picam2.start()
    time.sleep(1)

    threading.Thread(target=capture_loop, daemon=True).start()
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        picam2.stop()
