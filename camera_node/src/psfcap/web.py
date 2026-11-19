"""Flask debug view: live camera stream, run status and the last saved PSF."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2

from .analysis import to_display
from .camera import Camera
from .state import RunState

log = logging.getLogger(__name__)

INDEX_HTML = (Path(__file__).parent / "templates" / "index.html").read_text()


def start_web(camera: Camera, state: RunState, port: int) -> None:
    from flask import Flask, Response, jsonify
    from werkzeug.serving import make_server

    app = Flask("psf_capture")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            seq = 0
            while True:
                f = camera.wait_newer(seq, timeout=2.0)
                if f is None or f.seq == seq:
                    continue
                seq = f.seq
                img = to_display(f.raw, 0.75)
                z = "-" if state.distance_mm is None else f"{state.distance_mm:.3f}"
                txt = (f"{state.phase} {state.index}/{state.total}  z={z}mm  "
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
        d["camera"] = {"exposure_us": f.exposure_us if f else None,
                       "frame_max": int(f.raw.max()) if f else None, "error": camera.error}
        return jsonify(d)

    @app.get("/api/last_capture.jpg")
    def last_capture():
        if state.last_jpeg is None:
            return Response(status=204)
        return Response(state.last_jpeg, mimetype="image/jpeg")

    srv = make_server("0.0.0.0", port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, name="web", daemon=True).start()
    log.info("Debug web UI on http://%s:%d/", _local_ip(), port)


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
