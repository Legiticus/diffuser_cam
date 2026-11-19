#!/usr/bin/env python3
"""
Interactive PSF viewer for a psf_capture run.

Pick a point in the object space (x, y at depth d) and see the PSF the sensor should
record for it, predicted from the captured PSFs (see psf_model.py for the model).

    python3 app.py                              # latest run in camera_node/data/
    python3 app.py ../camera_node/data/psf_20261115_192500 --t-mm 6.2
    python3 synthetic.py /tmp/demo && python3 app.py /tmp/demo   # try without real data

Then open http://<pi>:8003/
"""

from __future__ import annotations

import argparse
import io
import math
import socket
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from psf_model import PIXEL_PITCH_UM, PsfModel, render

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "camera_node" / "data"


def latest_run(data_dir: Path) -> Path:
    runs = sorted(p.parent for p in data_dir.glob("psf_*/manifest.json"))
    if not runs:
        raise SystemExit(f"No runs with a manifest.json in {data_dir}; pass a run directory")
    return runs[-1]


def create_app(model: PsfModel) -> Flask:
    app = Flask(__name__)

    def params():
        a = request.args
        x = float(a.get("x", 0))
        y = float(a.get("y", 0))
        d = float(a.get("d", model.depth_range[0]))
        t = float(a["t"]) if a.get("t") else model.t_mm
        if not all(math.isfinite(v) for v in (x, y, d, t)) or t <= 0:
            raise ValueError("x, y, d and t must be finite numbers, t > 0")
        return x, y, d, t

    def png(bgr) -> Response:
        ok, buf = cv2.imencode(".png", bgr)
        return Response(buf.tobytes(), mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.errorhandler(ValueError)
    def bad_request(e):
        return jsonify(error=str(e)), 400

    @app.get("/")
    def index():
        return (Path(__file__).parent / "templates" / "index.html").read_text()

    @app.get("/api/info")
    def info():
        est = model.t_estimate
        return jsonify(
            run_dir=str(model.run_dir), t_mm=model.t_mm, pixel_pitch_um=model.pitch_mm * 1000,
            sensor_wh=[model.w, model.h], depth_range=model.depth_range,
            t_estimate={"t_mm": est.t_mm, "rms_residual": est.rms_residual, "note": est.note,
                        "scales": est.scales},
            captures=[{"index": p.index, "distance_mm": p.distance_mm, "file": p.file,
                       "exposure_us": p.exposure_us, "dark_subtracted": p.dark_subtracted,
                       "warnings": p.warnings} for p in model.psfs])

    @app.get("/api/predict.json")
    def predict_json():
        return jsonify(model.predict(*params()).info)

    @app.get("/api/predict.png")
    def predict_png():
        p = model.predict(*params())
        gamma = float(request.args.get("gamma", 0.5))
        return png(render(p.image, p.known, gamma=gamma, scale=0.6))

    @app.get("/api/predict.npy")
    def predict_npy():
        x, y, d, t = params()
        p = model.predict(x, y, d, t)
        bio = io.BytesIO()
        np.save(bio, p.image)
        name = f"psf_pred_x{x:g}_y{y:g}_z{d:g}mm.npy"
        return Response(bio.getvalue(), mimetype="application/octet-stream",
                        headers={"Content-Disposition": f"attachment; filename={name}"})

    @app.get("/api/fov.json")
    def fov_json():
        _, _, d, t = params()
        _, x_max, y_max = model.fov_map(d, t, nx=3, ny=3)
        return jsonify(x_max_mm=x_max, y_max_mm=y_max)

    @app.get("/api/fov.png")
    def fov_png():
        _, _, d, t = params()
        frac, _, _ = model.fov_map(d, t)
        img = cv2.applyColorMap((np.clip(frac, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        return png(cv2.resize(img, (480, 300), interpolation=cv2.INTER_LINEAR))

    @app.get("/api/captured/<int:i>.png")
    def captured_png(i: int):
        p = next((p for p in model.psfs if p.index == i), None)
        if p is None:
            return jsonify(error="no such capture"), 404
        return png(render(p.image, gamma=float(request.args.get("gamma", 0.5)), scale=0.6))

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
    p = argparse.ArgumentParser(description="Interactive PSF viewer for a psf_capture run")
    p.add_argument("run_dir", nargs="?", help=f"capture run folder (default: latest in {DEFAULT_DATA})")
    p.add_argument("--t-mm", type=float, help="diffuser-to-sensor distance (mm); estimated from the run if omitted")
    p.add_argument("--pixel-um", type=float, default=PIXEL_PITCH_UM, help="sensor pixel pitch (um)")
    p.add_argument("--port", type=int, default=8003)
    a = p.parse_args(argv)
    run = Path(a.run_dir) if a.run_dir else latest_run(DEFAULT_DATA)
    print(f"Loading {run}")
    model = PsfModel(run, a.t_mm, a.pixel_um)
    print(f"Viewer on http://{_local_ip()}:{a.port}/")
    create_app(model).run(host="0.0.0.0", port=a.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
