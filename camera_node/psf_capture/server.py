"""FastAPI app: REST API, server-sent events, MJPEG live view and the static frontend.

  GET  /                       frontend (static/index.html, Alpine.js, no build step)
  GET  /api/status, /api/events     state snapshot; SSE stream of snapshots on change
  GET  /stream.mjpg            low-resolution preview (viewing only, never saved)
  GET  /api/live/stats         histogram + saturation of the latest raw frame
  POST /api/stage/home, /api/stage/move, /api/camera/controls      (idle only)
  GET  /api/defaults           option defaults and hardware limits
  GET/POST/DELETE /api/plans[...], POST /api/plans/preview
  POST /api/session/start|pause|abort|continue|clear_error
  GET  /api/sessions
  POST /api/model/build, /api/model/generate;  GET /api/model/result.(npy|png|tiff|preview.png)
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import plans, processing, storage
from .camera import frame_stats
from .config import Z_HOME_MM, Z_MAX_MM
from .core import Backend, Busy
from .model import PsfModel
from .stage import StageError

STATIC = Path(__file__).parent / "static"


def create_app(backend: Backend) -> FastAPI:
    app = FastAPI(title="PSF Capture")
    b = backend
    model: dict = {"model": None, "result": None, "result_id": 0}

    @app.exception_handler(Busy)
    async def _busy(_, e):
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(ValueError)
    async def _bad(_, e):
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(StageError)
    async def _stage(_, e):
        return JSONResponse({"detail": f"Stage: {e}"}, status_code=500)

    # --- status ----------------------------------------------------------------------
    @app.get("/api/status")
    def status():
        return b.snapshot()

    @app.get("/api/events")
    async def events(request: Request):
        async def stream():
            last, last_sent = None, 0.0
            while not await request.is_disconnected():
                if b.version != last:
                    last = b.version
                    last_sent = time.monotonic()
                    yield f"data: {json.dumps(b.snapshot())}\n\n"
                elif time.monotonic() - last_sent > 15:
                    last_sent = time.monotonic()
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.2)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    # --- live view ----------------------------------------------------------------------
    @app.get("/stream.mjpg")
    async def mjpeg(request: Request):
        async def stream():
            while not await request.is_disconnected():
                frame = await run_in_threadpool(b.camera.preview_jpeg)
                if frame:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                await asyncio.sleep(0.1)
        return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/live/stats")
    def live_stats():
        frame = b.camera.latest()
        if frame is None:
            return {}
        return {**frame_stats(frame.raw, b.camera.black_level),
                "exposure_us": frame.exposure_us, "gain": frame.gain}

    @app.post("/api/stage/home")
    def home():
        b.home()
        return b.snapshot()

    @app.post("/api/stage/move")
    def move(body: dict):
        b.move(float(body["z"]))
        return b.snapshot()

    @app.post("/api/camera/controls")
    def controls(body: dict):
        return b.set_live_controls(float(body["exposure_us"]), float(body["gain"]))

    # --- capture --------------------------------------------------------------------------
    @app.get("/api/defaults")
    def defaults():
        return {"options": b.default_options(), "z_range": [Z_HOME_MM, Z_MAX_MM],
                "exposure_limits": b.camera.exposure_limits, "gain_limits": b.camera.gain_limits,
                "processing": processing.DEFAULT_STEPS}

    @app.get("/api/plans")
    def list_plans():
        return b.plans.all()

    @app.post("/api/plans")
    def save_plan(body: dict):
        return b.plans.save(body)

    @app.delete("/api/plans/{name}")
    def delete_plan(name: str):
        b.plans.delete(name)
        return b.plans.all()

    @app.post("/api/plans/preview")
    def preview_plan(body: dict):
        return plans.positions(body)

    @app.post("/api/session/start")
    def start(body: dict):
        return {"session_id": b.start_session(body)}

    @app.post("/api/session/pause")
    def pause(body: dict):
        b.pause(bool(body.get("paused", True)))
        return b.snapshot()

    @app.post("/api/session/abort")
    def abort():
        b.abort()
        return b.snapshot()

    @app.post("/api/session/continue")
    def cont():
        b.continue_prompt()
        return b.snapshot()

    @app.post("/api/session/clear_error")
    def clear_error():
        b.clear_error()
        return b.snapshot()

    # --- model ------------------------------------------------------------------------------
    @app.get("/api/sessions")
    def sessions():
        return storage.list_sessions(b.data_dir)

    @app.post("/api/model/build")
    def build(body: dict):
        ids = body.get("sessions") or []
        steps = processing.normalise_steps(body.get("steps"))
        if not ids:
            raise ValueError("Select at least one session")
        for sid in ids:
            if not (b.data_dir / sid / storage.METADATA).exists():
                raise ValueError(f"Unknown session {sid}")

        def job(report):
            dirs = []
            for n, sid in enumerate(ids):
                dirs.append(processing.process_session(
                    b.data_dir / sid, steps,
                    lambda f, msg: report((n + f) / (len(ids) + 1), msg)))
            m = PsfModel(dirs, lambda f, msg: report((len(ids) + f) / (len(ids) + 1), msg))
            model["model"] = m
            infos = [processing.load(d)[1] for d in dirs]
            return {"summary": m.summary(), "sessions": [
                {k: i[k] for k in ("session", "x_mm", "y_mm", "pass_shifts", "drift",
                                   "rail_slope", "warnings", "units", "no_psf", "z_mm")}
                for i in infos]}

        b.run_job(job)
        return b.snapshot()

    @app.post("/api/model/generate")
    def generate(body: dict):
        b.require_idle_for_model()
        if model["model"] is None:
            raise ValueError("Build a model first")
        points = body.get("points")
        if not isinstance(points, list):
            raise ValueError('Expected {"points": [{"x":..,"y":..,"z":..,"intensity":..}]}')
        img, warnings, details = model["model"].generate(points)
        model["result"] = np.clip(img, 0, None).astype(np.float32)    # final output: clip negatives
        model["result_id"] += 1
        r = model["result"]
        return {"id": model["result_id"], "warnings": warnings, "details": details,
                "max": float(r.max()), "sum": float(r.sum()),
                "scale_16bit": _scale16(r)}

    def result():
        if model["result"] is None:
            raise ValueError("Nothing generated yet")
        return model["result"]

    @app.get("/api/model/result.npy")
    def result_npy():
        buf = io.BytesIO()
        np.save(buf, result())
        return Response(buf.getvalue(), media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=psf.npy"})

    @app.get("/api/model/result.png")
    def result_png():
        buf = io.BytesIO()
        Image.fromarray(_to16(result())).save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png",
                        headers={"Content-Disposition": "attachment; filename=psf16.png"})

    @app.get("/api/model/result.tiff")
    def result_tiff():
        buf = io.BytesIO()
        tifffile.imwrite(buf, _to16(result()))
        return Response(buf.getvalue(), media_type="image/tiff",
                        headers={"Content-Disposition": "attachment; filename=psf16.tiff"})

    @app.get("/api/model/preview.png")
    def result_preview(gamma: float = 1.0):
        r = result()
        view = (r / r.max()) ** (1 / gamma) if r.max() > 0 else r
        buf = io.BytesIO()
        Image.fromarray((view * 255).astype(np.uint8), "L").save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})

    # --- frontend ---------------------------------------------------------------------------
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _scale16(img: np.ndarray) -> float:
    """Multiply the float result by this to get the 16-bit image (max -> 65535)."""
    return 65535.0 / float(img.max()) if img.max() > 0 else 1.0


def _to16(img: np.ndarray) -> np.ndarray:
    return np.round(img * _scale16(img)).astype(np.uint16)
