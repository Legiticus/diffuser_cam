"""One capture session: home, sweep z (repeated per pass), check homing, backgrounds.

Flow (runs in the backend's capture thread):
  1. validate options, create the session folder and metadata.json
  2. home the stage (the front switch is the only reference)
  3. for each pass, for each planned z: move (always arriving from below), settle,
     choose the exposure, take the frames, write each one immediately
  4. end-of-session homing check: where the switch triggers, compared with a probe
     taken right after the initial homing (a difference means steps were lost)
  5. background: prompt light off, frames at every exposure used, prompt light on
Raw frames are stored unchanged; all corrections happen later in processing.py.
"""

from __future__ import annotations

import math
import time
from datetime import datetime

from . import plans, storage
from .camera import robust_peak
from .config import (BACKLASH_MM, EXPOSURE_STEPS_PER_OCTAVE, HOME_TOLERANCE_MM, SATURATION,
                     Z_HOME_MM)

MAX_RETAKES = 6


def exposure_ladder(lo: int, hi: int) -> list[int]:
    """Allowed per-position exposures: EXPOSURE_STEPS_PER_OCTAVE per doubling, plus hi."""
    k0 = math.ceil(EXPOSURE_STEPS_PER_OCTAVE * math.log2(max(lo, 1)))
    rungs = set()
    k = k0
    while (e := round(2 ** (k / EXPOSURE_STEPS_PER_OCTAVE))) <= hi:
        if e >= lo:
            rungs.add(e)
        k += 1
    rungs.add(int(hi))
    return sorted(rungs)


def parse_options(params: dict, defaults: dict, exposure_limits: tuple[int, int]) -> dict:
    """Merge user input over defaults and check it. Raises ValueError."""
    o = {**defaults, **{k: v for k, v in params.items() if v is not None and v != ""}}
    for key in ("x_mm", "y_mm"):
        if key not in params or params[key] in (None, ""):
            raise ValueError(f"{key[0]} offset is required")
        o[key] = float(params[key])
    for key in ("n_positions", "frames_per_position", "passes", "allowed_saturated_px"):
        o[key] = int(o[key])
    o["background"] = bool(o["background"])
    o["background_frames"] = int(o.get("background_frames") or o["frames_per_position"])
    for key in ("target_fraction", "min_signal_fraction", "settle_s", "exposure_us", "gain",
                "max_exposure_us"):
        o[key] = float(o[key])
    if o["exposure_mode"] not in ("fixed", "per_position"):
        raise ValueError("exposure_mode must be 'fixed' or 'per_position'")
    if o["frames_per_position"] < 1 or o["passes"] < 1 or o["background_frames"] < 1:
        raise ValueError("Frames, passes and background frames must be at least 1")
    if not 0 < o["target_fraction"] < 1:
        raise ValueError("Exposure target must be between 0 and 1")
    if not 0 <= o["min_signal_fraction"] < o["target_fraction"]:
        raise ValueError("No-PSF threshold must be below the exposure target")
    if o["settle_s"] < 0:
        raise ValueError("Settle time cannot be negative")
    lo, hi = exposure_limits
    if not lo <= o["max_exposure_us"] <= hi:
        raise ValueError(f"Maximum exposure must be within {lo}..{hi} us")
    o["max_exposure_us"] = int(o["max_exposure_us"])
    return o


class CaptureSession:
    def __init__(self, backend, params: dict):
        self.b = backend
        cam = backend.camera
        self.opt = parse_options(params, backend.default_options(), cam.exposure_limits)
        self.plan = plans.positions({k: self.opt[k] for k in plans.PLAN_KEYS} |
                                    {"name": params.get("plan_name")})
        self.plan["name"] = self.plan["name"] or backend.plans.matching_name(self.plan)
        self.plan_warnings = self.plan.pop("warnings")
        self.started = datetime.now()
        self.id = storage.session_name(self.started, self.opt["x_mm"], self.opt["y_mm"])
        self.dir = backend.data_dir / self.id
        self.ladder = exposure_ladder(cam.exposure_limits[0], self.opt["max_exposure_us"])
        if self.opt["exposure_mode"] == "fixed":
            self.exposure = int(self.opt["exposure_us"])
        else:
            self.exposure = min(self.ladder, key=lambda e: abs(math.log(e / self.opt["exposure_us"])))
        self.gain = self.opt["gain"]
        self.meta: dict = {}
        self.t_sweep: float | None = None
        self.positions_done = 0
        self.frame_count = 0

    # --- bookkeeping ---------------------------------------------------------------
    def save(self) -> None:
        storage.write_json(self.dir / storage.METADATA, self.meta)

    def warn(self, msg: str) -> None:
        self.meta["warnings"].append(msg)
        self.b.warn(msg)

    def finish(self, status: str, error: str | None = None) -> None:
        if not self.meta:
            return
        self.meta.update(status=status, ended=datetime.now().isoformat(timespec="seconds"))
        if error:
            self.meta["error"] = error
        self.save()

    def report(self, **extra) -> None:
        total = self.opt["passes"] * self.opt["n_positions"]
        eta = None
        if self.t_sweep and self.positions_done:
            per = (time.monotonic() - self.t_sweep) / self.positions_done
            eta = per * (total - self.positions_done)
        self.b.changed(progress={**self.b.progress, "positions_done": self.positions_done,
                                 "positions_total": total, "passes": self.opt["passes"], "frames": self.frame_count,
                                 "eta_s": eta, "z_mm": self.b.stage.z, **extra})

    # --- the procedure ----------------------------------------------------------------
    def run(self) -> None:
        o, cam = self.opt, self.b.camera
        self.dir.mkdir(parents=True)
        self.meta = {
            "app": storage.app_info(),
            "status": "running",
            "x_mm": o["x_mm"], "y_mm": o["y_mm"],
            "plan": self.plan,
            "options": {k: o[k] for k in o if k not in ("x_mm", "y_mm")},
            "camera": {"sensor": "OV9281", "width": 1280, "height": 800, "bit_depth": 10,
                       "saturation_level": SATURATION, "black_level": cam.black_level,
                       "stored_as": "uint16 TIFF, raw values unchanged"},
            "started": self.started.isoformat(timespec="seconds"), "ended": None,
            "positions": [], "frames": [], "backgrounds": [], "homing_checks": [],
            "warnings": list(self.plan_warnings),
        }
        if self.b.world is not None:                       # fake: the user 'moved' the source
            self.b.world.x_mm, self.b.world.y_mm, self.b.world.light_on = o["x_mm"], o["y_mm"], True
        self.save()

        self.report(phase="homing")
        self.b.stage.home()
        self.meta["homing_baseline_mm"] = self.b.stage.probe_switch()   # where the switch presses now
        self.save()
        self.t_sweep = time.monotonic()
        for pass_no in range(1, o["passes"] + 1):
            if pass_no > 1 and self.plan["actual_z"][0] - BACKLASH_MM < Z_HOME_MM:
                self.homing_check(f"before pass {pass_no}")   # re-home so z_min is approached from below
            for i, z in enumerate(self.plan["actual_z"]):
                self.capture_position(pass_no, i, z)

        self.report(phase="homing check")
        self.homing_check("end of session")
        if o["background"]:
            self.capture_backgrounds()
        self.finish("complete")
        self.report(phase="done", eta_s=0)

    def homing_check(self, when: str) -> None:
        """Probe the switch again; if it now triggers at a different slider position
        than at the start of the session, steps were lost (the stepper is open loop)."""
        trigger = self.b.stage.probe_switch()
        baseline = self.meta["homing_baseline_mm"]
        offset = None if trigger is None or baseline is None else round(trigger - baseline, 4)
        result = {"when": when, "trigger_position_mm": trigger, "baseline_mm": baseline,
                  "offset_mm": offset, "tolerance_mm": HOME_TOLERANCE_MM,
                  "steps_lost": offset is None or abs(offset) > HOME_TOLERANCE_MM}
        self.meta["homing_checks"].append(result)
        self.meta["homing_check"] = result                 # the latest (end of session) result
        self.save()
        if result["steps_lost"]:
            self.warn(f"Homing check {when}: limit switch triggered {offset} mm away from where "
                      f"it triggered at the start of the session (steps were lost)")

    def capture_position(self, pass_no: int, i: int, z: float) -> None:
        o, cam = self.opt, self.b.camera
        self.b.checkpoint()
        self.report(phase="moving", pass_no=pass_no, z_index=i, z_target=z)
        reached = self.b.stage.move_to_z(z)
        self.b.sleep(o["settle_s"])
        settled_ns = time.monotonic_ns()

        no_psf = False
        if o["exposure_mode"] == "per_position":
            self.report(phase="exposing")
            no_psf = not self.auto_expose(settled_ns)
        else:
            cam.set_controls(self.exposure, self.gain)
        position = {"z_index": i, "pass": pass_no, "z_mm": reached, "no_psf": no_psf,
                    "requested_exposure_us": self.exposure, "gain": self.gain}
        if no_psf:
            self.warn(f"No PSF at z={reached:.3f} mm (pass {pass_no}): signal too weak at the "
                      f"maximum exposure {self.exposure} us")
        else:
            self.report(phase="capturing")
            for attempt in range(MAX_RETAKES):
                if self.take_frames(pass_no, i, reached, settled_ns, last_try=attempt == MAX_RETAKES - 1):
                    break
            position["requested_exposure_us"] = self.exposure
        self.meta["positions"].append(position)
        self.save()
        self.positions_done += 1
        self.report()

    def take_frames(self, pass_no: int, i: int, z: float, after_ns: int, last_try: bool) -> bool:
        """Take and store the frames for one position. Returns False if they must be
        retaken at a lower exposure (per-position mode, saturated)."""
        o, cam = self.opt, self.b.camera
        cam.set_controls(self.exposure, self.gain)
        entries, warned = [], False
        for f in range(o["frames_per_position"]):
            self.b.checkpoint()
            frame = cam.grab(1, after_ns)[0]
            after_ns = frame.timestamp_ns
            saturated = int((frame.raw >= SATURATION).sum())
            if saturated > o["allowed_saturated_px"]:
                retake = o["exposure_mode"] == "per_position" and self.exposure > self.ladder[0]
                if retake and not last_try:
                    for e in entries:                    # discard this position's frames, retake
                        (self.dir / e["file"]).unlink(missing_ok=True)
                        self.meta["frames"].remove(e)
                    self.frame_count -= len(entries)
                    self.exposure = self.step_down(self.exposure)
                    self.save()
                    return False
                if not warned:
                    self.warn(f"{saturated} saturated pixels at z={z:.3f} mm, pass {pass_no}, "
                              f"frame {f} ({self.exposure} us)")
                    warned = True
            name = storage.frame_name(i, pass_no, f)
            storage.write_frame(self.dir / name, frame.raw)
            entry = {"file": name, "z_index": i, "pass": pass_no, "frame": f, "z_mm": z,
                     "requested_exposure_us": self.exposure, "exposure_us": frame.exposure_us,
                     "gain": frame.gain, "timestamp_ns": frame.timestamp_ns,
                     "saturated_px": saturated, "peak": robust_peak(frame.raw)}
            entries.append(entry)
            self.meta["frames"].append(entry)
            self.frame_count += 1
            self.save()
            self.report()
        return True

    def step_down(self, e: int) -> int:
        lower = [r for r in self.ladder if r < e]
        return lower[-1] if lower else self.ladder[0]

    def auto_expose(self, after_ns: int) -> bool:
        """Pick the longest ladder exposure whose brightest pixel stays below the target.
        Returns False ('no PSF') if even the maximum exposure gives too little signal."""
        o, cam = self.opt, self.b.camera
        full = SATURATION - cam.black_level
        target = o["target_fraction"] * full
        peak = 0.0
        for _ in range(12):
            self.b.checkpoint()
            cam.set_controls(self.exposure, self.gain)
            frame = cam.grab(1, after_ns)[0]
            after_ns = frame.timestamp_ns
            peak = robust_peak(frame.raw) - cam.black_level
            saturated = int((frame.raw >= SATURATION).sum())
            if saturated > o["allowed_saturated_px"] or peak >= 0.98 * full:
                best = max(self.ladder[0], self.exposure // 4)   # true brightness unknown: big step
                best = max([r for r in self.ladder if r <= best], default=self.ladder[0])
                if best == self.exposure:
                    return True                                # cannot go lower; frames will warn
            else:
                wanted = self.exposure * min(16.0, target / max(peak, 0.5))
                best = max([r for r in self.ladder if r <= wanted], default=self.ladder[0])
            if best == self.exposure:
                break
            self.exposure = best
        self.report(exposure_us=self.exposure)
        return not (self.exposure == self.ladder[-1] and peak < o["min_signal_fraction"] * full)

    def capture_backgrounds(self) -> None:
        o, cam = self.opt, self.b.camera
        exposures = sorted({f["requested_exposure_us"] for f in self.meta["frames"]})
        if not exposures:
            return
        self.report(phase="background")
        self.b.ask_user("light_off", "Turn the light source OFF, then press Continue to take "
                                     f"background frames at {len(exposures)} exposure time(s).")
        self.report(phase="background")
        for e in exposures:
            cam.set_controls(e, self.gain)
            after_ns = time.monotonic_ns()
            for f in range(o["background_frames"]):
                self.b.checkpoint()
                frame = cam.grab(1, after_ns)[0]
                after_ns = frame.timestamp_ns
                name = storage.background_name(e, f)
                storage.write_frame(self.dir / name, frame.raw)
                self.meta["backgrounds"].append({
                    "file": name, "requested_exposure_us": e, "frame": f,
                    "exposure_us": frame.exposure_us, "gain": frame.gain,
                    "timestamp_ns": frame.timestamp_ns})
                self.frame_count += 1
                self.save()
                self.report()
        self.b.ask_user("light_on", "Background frames done. Turn the light source back ON, "
                                    "then press Continue.")
