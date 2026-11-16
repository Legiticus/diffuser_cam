"""
Keeps the viewer on the most recent capture run.

A background thread polls the data folder. Whenever a newer run appears, or the
current run's manifest changes (a new PSF during a sweep, dark frames at the end),
the model is rebuilt in the background and swapped in; the old one keeps serving
until then. Only files that changed are reloaded.

Estimating t takes a while on the Pi, so estimates are cached on disk
(~/.cache/diffuser_cam/psf_viewer/geometry.json) and reused when the viewer restarts.
While a run is too short for a reliable estimate (e.g. the first PSFs of a sweep), the
latest reliable estimate from an earlier run is used instead, and the page says so.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path

from psf_model import (PIXEL_PITCH_UM, Capture, Geometry, PsfModel, capture_key,
                       estimate_geometry, load_captures, read_manifest, sample_key)

DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "diffuser_cam" / "psf_viewer"


def find_latest_run(data_dir: Path) -> Path | None:
    """Newest run folder (psf_YYYYMMDD_HHMMSS, so name order is time order) with a manifest."""
    runs = sorted(p.parent for p in Path(data_dir).glob("psf_*/manifest.json"))
    return runs[-1] if runs else None


class GeometryCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path and path.exists():
            try:
                self.entries = json.loads(path.read_text())
            except (OSError, ValueError):
                self.entries = {}

    def get(self, run: Path, key: list) -> Geometry | None:
        e = self.entries.get(str(run))
        return Geometry.from_json(e["geometry"]) if e and e.get("key") == key else None

    def put(self, run: Path, key: list, geo: Geometry):
        self.entries[str(run)] = {"key": key, "geometry": geo.to_json(), "time": time.time()}
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.entries, indent=1, default=float))
                tmp.replace(self.path)
            except OSError:
                pass

    def latest_good(self, exclude: Path) -> tuple[str, Geometry] | None:
        best = None
        for run, e in self.entries.items():
            g = Geometry.from_json(e["geometry"])
            if run != str(exclude) and g.ok and (best is None or e.get("time", 0) > best[0]):
                best = (e.get("time", 0), run, g)
        return (best[1], best[2]) if best else None


class RunWatcher:
    def __init__(self, data_dir: Path | None = None, run_dir: Path | None = None,
                 t_mm: float | None = None, pixel_pitch_um: float = PIXEL_PITCH_UM,
                 poll_s: float = 2.0, cache_path: Path | None = DEFAULT_CACHE / "geometry.json",
                 n_samples: int = 6, log=print):
        if (data_dir is None) == (run_dir is None):
            raise ValueError("give exactly one of data_dir (follow latest) or run_dir (fixed run)")
        self.data_dir = Path(data_dir) if data_dir else None
        self.fixed_run = Path(run_dir) if run_dir else None
        self.t_override = t_mm
        self.pitch = pixel_pitch_um
        self.poll_s = poll_s
        self.n_samples = n_samples
        self.log = log
        self.geo_cache = GeometryCache(cache_path)
        self._caps: dict[tuple, Capture] = {}
        self._caps_run: Path | None = None
        self._sig = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.model: PsfModel | None = None
        self.version = 0
        self.status = {"phase": "starting", "message": "", "run": None, "run_status": None,
                       "n_psfs": 0, "updated": None}

    # ------------------------------------------------------------------ state
    def _set(self, **kw):
        with self._lock:
            self.status = {**self.status, **kw}

    def snapshot(self) -> dict:
        with self._lock:
            m = self.model
            return {**self.status, "version": self.version,
                    "following_latest": self.fixed_run is None,
                    "model_run": str(m.run_dir) if m else None,
                    "model_n_psfs": len(m.caps) if m else 0}

    # ------------------------------------------------------------------ polling
    def _signature(self):
        run = self.fixed_run or find_latest_run(self.data_dir)
        if run is None:
            return None
        try:
            st = (run / "manifest.json").stat()
        except FileNotFoundError:
            return None
        return (str(run), st.st_mtime_ns, st.st_size)

    def poll_once(self) -> bool:
        """Check for new data; rebuild the model if there is any. Returns True if it changed."""
        sig = self._signature()
        if sig is None:
            where = self.fixed_run or self.data_dir
            self._set(phase="waiting", message=f"no capture runs in {where} yet")
            return False
        if sig == self._sig:
            return False
        run = Path(sig[0])
        try:
            self._load(run)
        except Exception as e:  # noqa: BLE001 - keep serving the previous model
            self.log(f"Failed to load {run}: {e}")
            traceback.print_exc()
            self._set(phase="error", message=f"{run.name}: {e}", run=str(run))
        self._sig = sig       # don't retry a failing manifest until it changes again
        return True

    def _resolve(self, run: Path, est: Geometry | None) -> Geometry | None:
        """The geometry to use, given this run's own estimate (None = not estimated yet)."""
        if self.t_override is not None:
            centre = est.center_px if est else None
            if centre is None and self.model is not None and self.model.run_dir == run:
                centre = self.model.center
            return Geometry(self.t_override, centre, None, "ok", "given (--t-mm)")
        if est is not None and est.ok:
            return est
        if est is None and self.model is not None and self.model.run_dir == run \
                and self.model.geometry.t_mm is not None:
            return self.model.geometry       # still re-estimating: keep this run's last one
        prev = self.geo_cache.latest_good(exclude=run)
        why = est.note if est is not None else "estimate pending"
        if prev:
            prun, pgeo = prev
            pgeo.note = f"this run: {why}; using the estimate from {Path(prun).name}"
            pgeo.source = Path(prun).name
            return pgeo
        return est        # unreliable or missing; PsfModel refuses if t is None

    def _publish(self, run: Path, manifest: dict, caps: list[Capture], geo: Geometry | None,
                 pending: bool):
        if geo is None or geo.t_mm is None:
            return
        model = PsfModel(run, manifest, caps, geo, self.pitch)
        with self._lock:
            self.model = model
            self.version += 1
            self.status = {**self.status, "updated": time.strftime("%H:%M:%S"),
                           **({} if pending else {"phase": "ready", "message": ""})}
        self.log(f"Showing {run.name}: {len(caps)} PSFs, status {manifest.get('status')}, "
                 f"t = {model.t_mm:.3f} mm ({geo.source}){' - re-estimating' if pending else ''}")

    def _load(self, run: Path):
        if run != self._caps_run:
            self._caps.clear()
            self._caps_run = run
        manifest = read_manifest(run)
        n = len(manifest.get("psfs", []))
        self._set(phase="loading", message=f"loading {run.name} ({n} PSFs)…", run=str(run),
                  run_status=manifest.get("status"), n_psfs=n)
        if n == 0:
            self._set(phase="waiting", message=f"{run.name}: no PSFs captured yet")
            return
        caps = load_captures(run, manifest, self._caps)
        live = {capture_key(r) for r in manifest["psfs"]}   # drop superseded entries
        for k in [k for k in self._caps if k not in live]:
            del self._caps[k]

        key = sample_key(caps, self.n_samples)
        est = self.geo_cache.get(run, key)
        if est is None and len(caps) >= 2:
            # estimating takes a while on the Pi: show the new data now with the best
            # geometry already known, then swap in the fresh estimate
            self._set(phase="loading", message=f"{run.name}: estimating t from {len(caps)} PSFs…")
            self._publish(run, manifest, caps, self._resolve(run, None), pending=True)
            self.log(f"Estimating t for {run.name} ({len(caps)} PSFs)")
            est = estimate_geometry(caps, self.n_samples, log=self.log)
            self.log(f"  t = {est.t_mm:.3f} mm, centre ({est.center_px[0]:.0f}, "
                     f"{est.center_px[1]:.0f}) px: {est.note}")
            self.geo_cache.put(run, key, est)
        elif est is None:
            est = Geometry(None, None, None, "need at least 2 PSFs to estimate t", "this run")
        geo = self._resolve(run, est)
        if geo is None or geo.t_mm is None:
            raise ValueError(f"t unknown ({est.note}) and no earlier estimate; pass --t-mm")
        self._publish(run, manifest, caps, geo, pending=False)

    # ------------------------------------------------------------------ thread
    def start(self) -> "RunWatcher":
        def loop():
            while not self._stop.is_set():
                self.poll_once()
                self._stop.wait(self.poll_s)
        self._thread = threading.Thread(target=loop, name="run-watcher", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
