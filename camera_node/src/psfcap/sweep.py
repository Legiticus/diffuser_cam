"""The PSF depth sweep: move, settle, auto-expose, capture, save, validate."""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .analysis import psf_stats, to_display
from .camera import Camera, Frame
from .constants import REPO_ROOT, SENSOR_MAX
from .controller import ControllerError, LimitHit, StageController
from .state import Aborted, RunState

log = logging.getLogger(__name__)


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
