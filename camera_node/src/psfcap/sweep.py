"""The PSF depth sweep: move, settle, auto-expose, capture, save, validate."""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .analysis import psf_stats, to_display
from .camera import Camera
from .constants import (DARK_FRAMES, GAIN, INITIAL_EXPOSURE_US, MAX_EXPOSURE_US, MAX_TRAVEL_MM,
                        MIN_EXPOSURE_US, MIN_SNR, SENSOR_MAX, SETTLE_S, STEPS_PER_MM, TARGET_PEAK,
                        ZERO_DISTANCE_MM)
from .controller import ControllerError, LimitHit, StageController
from .state import RunState

log = logging.getLogger(__name__)


def plan_distances(start: float, stop: float, num: int) -> list[float]:
    """Distances uniform in 1/z (the PSF scales with 1 + t/z), snapped to whole
    motor steps, sorted outward so backlash is always taken up the same way."""
    if num < 1:
        raise SystemExit("--num must be at least 1")
    d = [start] if num == 1 else list(1.0 / np.linspace(1.0 / start, 1.0 / stop, num))
    snapped = sorted({round((x - ZERO_DISTANCE_MM) * STEPS_PER_MM) for x in d})
    out = [ZERO_DISTANCE_MM + s / STEPS_PER_MM for s in snapped]
    for x in out:
        if x < ZERO_DISTANCE_MM - 1e-9:
            raise SystemExit(f"Distance {x:.3f} mm is closer than the zero position "
                             f"({ZERO_DISTANCE_MM} mm)")
        if x - ZERO_DISTANCE_MM > MAX_TRAVEL_MM + 1e-9:
            raise SystemExit(f"Distance {x:.3f} mm is beyond the maximum travel: the farthest "
                             f"allowed is {ZERO_DISTANCE_MM + MAX_TRAVEL_MM:.3f} mm "
                             f"(zero {ZERO_DISTANCE_MM} + {MAX_TRAVEL_MM} mm travel)")
    return out


class PsfSweep:
    def __init__(self, camera: Camera, stage: StageController, state: RunState,
                 run_dir: Path, frames: int):
        self.cam = camera
        self.stage = stage
        self.state = state
        self.dir = run_dir
        self.frames = frames
        self.records: list[dict] = []
        self.status = "running"
        self.meta = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "sensor": "OV9281 mono",
            "raw_size_wh": list(camera.raw_cfg["size"]),
            "data_units": "raw DN (0..1023), NOT dark-subtracted, mean of n_frames",
            "zero_distance_mm": ZERO_DISTANCE_MM,
            "steps_per_mm": STEPS_PER_MM,
            "analogue_gain": GAIN,
            "frames_per_psf": frames,
        }

    def auto_expose(self, start_exp: int, not_before_ns: int) -> int:
        """Find the exposure that puts the PSF's brightest pixels at ~TARGET_PEAK of
        full scale without saturating. Bracketing search: never returns to an exposure
        already seen saturating, and always leaves the camera set to the result."""
        lo_frac, hi_frac = TARGET_PEAK - 0.15, min(TARGET_PEAK + 0.15, 0.95)
        exp = int(np.clip(start_exp, MIN_EXPOSURE_US, MAX_EXPOSURE_US))
        ok_exp, sat_exp = None, None      # highest unsaturated / lowest saturated seen
        for it in range(12):
            self.cam.set_exposure(exp)
            self.state.exposure_us = exp
            f = self.cam.grab(1, not_before_ns)[0]
            img = f.raw.astype(np.float32)
            bg = float(np.percentile(img[::4, ::4], 1))
            peak = float(np.percentile(img, 99.999))   # ~10 brightest px, ignores lone hot px
            nsat = int((f.raw >= SENSOR_MAX).sum())
            frac = (peak - bg) / (SENSOR_MAX - bg)
            log.debug("  AE it%d exp=%dus peak=%.0f bg=%.0f frac=%.2f sat=%d", it, exp, peak, bg, frac, nsat)
            if nsat:
                sat_exp = exp if sat_exp is None else min(sat_exp, exp)
                new = exp * 0.4 if ok_exp is None else math.sqrt(ok_exp * sat_exp)
            else:
                ok_exp = exp if ok_exp is None else max(ok_exp, exp)
                if lo_frac <= frac <= hi_frac:
                    break
                new = exp * TARGET_PEAK / max(frac, 1e-3)
                if sat_exp is not None and new >= sat_exp * 0.97:
                    new = math.sqrt(exp * sat_exp)
            new = int(np.clip(new, MIN_EXPOSURE_US, MAX_EXPOSURE_US))
            if sat_exp is not None and ok_exp is not None and sat_exp <= ok_exp * 1.05:
                break       # bracket closed: ok_exp is as bright as we can go
            if new == exp:
                break       # pinned at an exposure limit; flagged later
            exp = new
        best = ok_exp if ok_exp is not None else exp
        self.cam.set_exposure(best)
        self.state.exposure_us = best
        return best

    def capture_at(self, exp: int, not_before_ns: int) -> tuple[np.ndarray, dict, int]:
        for attempt in range(3):
            exp = self.auto_expose(exp, not_before_ns)
            frames = self.cam.grab(self.frames, not_before_ns)
            acc = np.zeros(frames[0].raw.shape, np.float64)
            sat_max = 0
            for f in frames:
                acc += f.raw
                sat_max = max(sat_max, int((f.raw >= SENSOR_MAX).sum()))
            mean = (acc / len(frames)).astype(np.float32)
            if sat_max and attempt < 2:
                log.warning("  %d saturated px during averaging, reducing exposure", sat_max)
                exp = int(exp * 0.6)
                continue
            break
        info = {"exposure_us": frames[0].exposure_us, "requested_exposure_us": exp,
                "n_frames": len(frames), "saturated_px_max": sat_max}
        return mean, info, exp

    def save_psf(self, i: int, z: float, mean: np.ndarray, info: dict) -> dict:
        stem = f"psf_{i:03d}_z{z:08.3f}mm"
        np.save(self.dir / f"{stem}.npy", mean)
        st = psf_stats(mean)
        warns = []
        if info["saturated_px_max"]:
            warns.append(f"saturated ({info['saturated_px_max']} px)")
        if st["snr_peak"] < MIN_SNR:
            warns.append(f"low SNR ({st['snr_peak']:.0f})")
        if (st["peak"] - st["background"]) / (SENSOR_MAX - st["background"]) < 0.25:
            warns.append("peak < 25% of full scale")
        if st["edge_energy_frac"] is not None and st["edge_energy_frac"] > 0.02:
            warns.append(f"PSF energy at sensor edge ({100 * st['edge_energy_frac']:.1f}%) - may be cropped")
        if info["requested_exposure_us"] >= MAX_EXPOSURE_US:
            warns.append("at max exposure")
        rec = {
            "index": i,
            "distance_mm": round(z, 4),
            "travel_steps": self.stage.position_steps,
            "file": f"{stem}.npy",
            **info,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()},
            "warnings": warns,
        }
        for w in warns:
            log.warning("  z=%.3f mm: %s", z, w)
            self.state.warnings.append(f"z={z:.3f}: {w}")
        ok, jpg = cv2.imencode(".jpg", to_display(mean.astype(np.uint16), 0.5))
        self.state.last_jpeg = jpg.tobytes() if ok else None
        self.state.last_capture = rec
        return rec

    def write_manifest(self):
        m = dict(self.meta, status=self.status, psfs=self.records,
                 validation=self.validate(), warnings=self.state.warnings)
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(m, indent=2, default=str))
        tmp.replace(self.dir / "manifest.json")

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

    def run(self, distances: list[float]):
        st = self.state
        st.total = len(distances)
        st.phase = "zeroing"
        log.info("Zeroing stage...")
        self.stage.zero()
        st.distance_mm = ZERO_DISTANCE_MM
        exp = INITIAL_EXPOSURE_US
        prev_z = None
        try:
            for i, z in enumerate(distances):
                st.index = i + 1
                target = self.stage.mm_to_steps(z - ZERO_DISTANCE_MM)
                st.phase = "moving"
                log.info("[%d/%d] -> %.3f mm (travel %.4f mm, %d steps)",
                         i + 1, len(distances), z, z - ZERO_DISTANCE_MM, target)
                from_zero = self.stage.position_steps == 0
                try:
                    self.stage.move_to_steps(target)
                except LimitHit as e:
                    if e.which == "front" and from_zero:
                        raise ControllerError(
                            f"{e} on the first outward move: the travel direction looks inverted. "
                            f"Flip AWAY_SIGN in psfcap/constants.py.") from e
                    raise
                st.distance_mm = ZERO_DISTANCE_MM + self.stage.travel_mm
                st.phase = "settling"
                time.sleep(SETTLE_S)
                not_before = time.monotonic_ns()
                st.phase = "capturing"
                if prev_z is not None:
                    exp = int(exp * (z / prev_z) ** 2)   # intensity ~ 1/z^2, good AE start
                mean, info, exp = self.capture_at(exp, not_before)
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
            log.error("Limit switch hit at %d/%d: %s", st.index, st.total, e)
            if e.which == "back":
                log.error("The back switch tripped inside the %.0f mm software limit: the rail "
                          "may be shorter than MAX_TRAVEL_MM, or steps were lost.", MAX_TRAVEL_MM)
        except KeyboardInterrupt:
            self.status = "interrupted"
            raise
        finally:
            self.write_manifest()

    def capture_darks(self):
        exps = sorted({r["requested_exposure_us"] for r in self.records})
        if not exps:
            return
        self.state.phase = "waiting for dark"
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
            self.cam.set_exposure(exp)
            frames = self.cam.grab(DARK_FRAMES, time.monotonic_ns())
            mean = np.mean([f.raw for f in frames], axis=0, dtype=np.float64).astype(np.float32)
            name = f"dark_exp{exp:08d}us.npy"
            np.save(self.dir / name, mean)
            for r in self.records:
                if r["requested_exposure_us"] == exp:
                    r["dark_file"] = name
            log.info("  dark %s mean=%.1f", name, float(mean.mean()))
            if mean.max() > 0.5 * SENSOR_MAX:
                self.state.warnings.append(f"dark at {exp}us has bright pixels "
                                           f"(max {mean.max():.0f}) - was the source off?")
        self.write_manifest()
