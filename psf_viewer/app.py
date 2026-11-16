#!/usr/bin/env python3
"""
PSF viewer: shows the most recent psf_capture run and predicts the sensor PSF for
any point (x, y, d) in front of the camera (see psf_model.py for how).

    python3 app.py                       # follow the newest run in camera_node/data/
    python3 app.py --run ../camera_node/data/psf_20261116_111557   # one fixed run
    python3 app.py --t-mm 2.6            # use a known diffuser-to-sensor distance

Then open http://<pi>:8003/. The page switches to a new run (or new PSFs in a run
being captured) by itself.
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import socket
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from psf_model import PIXEL_PITCH_UM, PsfModel, render
from runs import RunWatcher

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "camera_node" / "data"


class NotReady(Exception):
    pass


def create_app(watcher: RunWatcher) -> Flask:
    app = Flask(__name__)

    def current() -> PsfModel:
        m = watcher.model
        if m is None:
            raise NotReady(watcher.snapshot()["message"] or "no data loaded yet")
        return m

    def params(m: PsfModel):
        a = request.args
        try:
            x = float(a.get("x", 0))
            y = float(a.get("y", 0))
            d = float(a.get("d", m.depth_range[0]))
            t = float(a["t"]) if a.get("t") else None
        except ValueError:
            raise ValueError("x, y, d and t must be numbers") from None
        if not all(math.isfinite(v) for v in (x, y, d)):
            raise ValueError("x, y and d must be finite numbers")
        return x, y, d, t

    def png(bgr) -> Response:
        ok, buf = cv2.imencode(".png", bgr)
        return Response(buf.tobytes(), mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.errorhandler(ValueError)
    def bad_request(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(NotReady)
    def not_ready(e):
        return jsonify(error=str(e), state=watcher.snapshot()), 503

    @app.get("/")
    def index():
        return (Path(__file__).parent / "templates" / "index.html").read_text()

    @app.get("/api/state")
    def state():
        return jsonify(watcher.snapshot())

    @app.get("/api/info")
    def info():
        m = current()
        g = m.geometry
        curve = m.curve(m.t_mm)
        return jsonify(
            version=watcher.version, run_dir=str(m.run_dir), run_name=m.run_dir.name,
            run_status=m.manifest.get("status"), created=m.manifest.get("created"),
            sensor=m.manifest.get("sensor"), t_mm=m.t_mm, scale_center_px=list(m.center),
            pixel_pitch_um=m.pitch_mm * 1000, sensor_wh=[m.w, m.h], depth_range=m.depth_range,
            geometry={"t_mm": g.t_mm, "note": g.note, "source": g.source,
                      "rms_residual": g.rms_residual, "ok": g.ok},
            energy_fit_max_residual=float(np.abs(curve.residual).max()),
            run_warnings=m.manifest.get("warnings", []),
            captures=[{"index": c.index, "distance_mm": c.distance_mm, "file": c.file,
                       "exposure_us": c.exposure_us, "dark_subtracted": c.dark_subtracted,
                       "warnings": c.warnings} for c in m.caps])

    @app.get("/api/predict.json")
    def predict_json():
        m = current()
        return jsonify(m.predict(*params(m)).info)

    @app.get("/api/predict.png")
    def predict_png():
        m = current()
        p = m.predict(*params(m))
        return png(render(p.image, p.known, gamma=float(request.args.get("gamma", 0.5))))

    @app.get("/api/predict.npy")
    def predict_npy():
        m = current()
        x, y, d, t = params(m)
        p = m.predict(x, y, d, t)
        bio = io.BytesIO()
        np.save(bio, p.image)
        name = f"psf_pred_{m.run_dir.name}_x{x:g}_y{y:g}_d{d:g}mm.npy"
        return Response(bio.getvalue(), mimetype="application/octet-stream",
                        headers={"Content-Disposition": f"attachment; filename={name}"})

    @app.get("/api/fov.json")
    def fov_json():
        m = current()
        _, _, d, t = params(m)
        _, x_max, y_max = m.fov_map(d, t, nx=3, ny=3)
        return jsonify(x_max_mm=x_max, y_max_mm=y_max)

    @app.get("/api/fov.png")
    def fov_png():
        m = current()
        _, _, d, t = params(m)
        frac, _, _ = m.fov_map(d, t)
        img = cv2.applyColorMap((frac * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        return png(cv2.resize(img, (480, 300), interpolation=cv2.INTER_LINEAR))

    @app.get("/api/captured/<int:i>.png")
    def captured_png(i: int):
        m = current()
        c = next((c for c in m.caps if c.index == i), None)
        if c is None:
            return jsonify(error="no such capture"), 404
        return png(render(c.image, gamma=float(request.args.get("gamma", 0.5))))

    return app


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PSF viewer for psf_capture runs")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA),
                   help="folder psf_capture writes runs to; the newest run is shown (default: %(default)s)")
    p.add_argument("--run", help="show this one run folder instead of following the newest")
    p.add_argument("--t-mm", type=float, help="diffuser-to-sensor distance (mm); estimated from the run if omitted")
    p.add_argument("--pixel-um", type=float, default=PIXEL_PITCH_UM, help="sensor pixel pitch (um)")
    p.add_argument("--poll", type=float, default=2.0, help="seconds between checks for new data")
    p.add_argument("--port", type=int, default=8003)
    a = p.parse_args(argv)
    if a.t_mm is not None and not a.t_mm > 0:
        p.error("--t-mm must be > 0")
    watcher = RunWatcher(data_dir=None if a.run else a.data_dir, run_dir=a.run, t_mm=a.t_mm,
                         pixel_pitch_um=a.pixel_um, poll_s=a.poll).start()
    print(f"Watching {a.run or a.data_dir}")
    print(f"Viewer on http://{_local_ip()}:{a.port}/")
    # the page polls /api/state every 2 s; don't log every request
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    create_app(watcher).run(host="0.0.0.0", port=a.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
