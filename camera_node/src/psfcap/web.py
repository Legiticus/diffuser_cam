"""Flask debug UI / API: live view, status, pause/resume/abort, preview stage control."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .analysis import to_display
from .camera import Camera
from .controller import StageController
from .state import RunState

log = logging.getLogger(__name__)


INDEX_HTML = (Path(__file__).parent / "templates" / "index.html").read_text()


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
