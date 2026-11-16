"""Command-line interface and program entry point."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

from .camera import Camera
from .constants import MAX_TRAVEL_MM, REPO_ROOT, SENSOR_MAX
from .controller import ControllerError, SimulatedSerial, StageController, find_serial_port
from .planning import plan_distances
from .state import RunState
from .sweep import PsfSweep
from .web import start_web

log = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DiffuserCam PSF depth-sweep capture",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("controller")
    g.add_argument("--port", default="auto", help="serial port, 'auto', or 'sim' for a simulated controller")
    g.add_argument("--baud", type=int, default=115200)
    g.add_argument("--zero-distance", type=float, default=38.0, help="source-to-sensor distance (mm) at ZERO")
    g.add_argument("--steps-per-mm", type=float, default=800.0, help="must match firmware STEPS_PER_MM")
    g.add_argument("--away-sign", type=int, choices=(-1, 1), default=-1,
                   help="sign of the MOVE argument that moves the source AWAY from the sensor")
    g.add_argument("--max-travel", type=float, default=MAX_TRAVEL_MM,
                   help=f"software limit on travel from zero (mm); can only be lowered below {MAX_TRAVEL_MM}")
    g.add_argument("--zero-timeout", type=float, default=None,
                   help="seconds; default allows homing from full travel at --move-speed")
    g.add_argument("--sim-back-limit", action="store_true",
                   help="with --port sim, emulate a back limit switch (not fitted on the real rig)")
    g.add_argument("--move-speed", type=float, default=0.5,
                   help="conservative speed (mm/s) used only to size move timeouts")
    g.add_argument("--rezero-each", action="store_true",
                   help="home before every position (slower; no accumulated open-loop error)")
    g.add_argument("--no-home", action="store_true", help="don't return to zero at the end")

    g = p.add_argument_group("distances")
    g.add_argument("--start", type=float, default=38.0, help="first distance (mm)")
    g.add_argument("--stop", type=float,
                   help=f"last distance (mm); at most zero-distance + max-travel ({38 + MAX_TRAVEL_MM:.0f} by default)")
    g.add_argument("--num", type=int, default=20, help="number of distances")
    g.add_argument("--spacing", choices=("inverse", "linear"), default="inverse",
                   help="inverse = uniform in 1/z (matches PSF scaling)")
    g.add_argument("--distances", help="explicit comma-separated list in mm (overrides start/stop/num)")

    g = p.add_argument_group("camera")
    g.add_argument("--frames", type=int, default=10, help="frames averaged per PSF")
    g.add_argument("--settle", type=float, default=1.0, help="seconds to wait after a move")
    g.add_argument("--exposure-us", type=int, default=None, help="fixed exposure (disables auto exposure)")
    g.add_argument("--initial-exposure-us", type=int, default=5000)
    g.add_argument("--min-exposure-us", type=int, default=20)
    g.add_argument("--max-exposure-us", type=int, default=500000)
    g.add_argument("--gain", type=float, default=1.0, help="analogue gain (1.0 = lowest noise)")
    g.add_argument("--target-peak", type=float, default=0.75, help="AE target peak as fraction of full scale")
    g.add_argument("--sat-level", type=int, default=SENSOR_MAX, help="raw value counted as saturated")
    g.add_argument("--max-saturated", type=int, default=0, help="saturated pixels tolerated per frame")
    g.add_argument("--min-snr", type=float, default=50.0, help="warn below this peak SNR")
    g.add_argument("--dark-frames", type=int, default=30)
    g.add_argument("--no-dark", action="store_true", help="skip the dark-frame step")

    g = p.add_argument_group("output / web")
    g.add_argument("--out-dir", default=str(REPO_ROOT / "camera_node" / "data"))
    g.add_argument("--host", default="0.0.0.0")
    g.add_argument("--http-port", type=int, default=8002)
    g.add_argument("--no-web", action="store_true")
    g.add_argument("--preview", action="store_true",
                   help="camera + web UI only (stage via /api/zero, /api/goto if a controller is found)")
    g.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    g.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if not 0 < args.max_travel <= MAX_TRAVEL_MM:
        p.error(f"--max-travel must be in (0, {MAX_TRAVEL_MM}] mm")
    if args.zero_timeout is None:
        args.zero_timeout = 30.0 + args.max_travel / args.move_speed
    return args


def make_stage(args, required: bool) -> StageController | None:
    try:
        port = find_serial_port() if args.port == "auto" else args.port
    except ControllerError:
        if required:
            raise
        log.warning("No controller found; stage control disabled")
        return None
    stage = StageController(port, args.baud, args.steps_per_mm, args.away_sign,
                            args.zero_timeout, args.move_speed, args.max_travel)
    if port == "sim":
        stage._ser = SimulatedSerial(args.steps_per_mm, back_limit_switch=args.sim_back_limit)
    else:
        stage.open()
    return stage


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("picamera2", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    distances = None if args.preview else plan_distances(args)
    if distances:
        print(f"\nPlanned {len(distances)} PSF distances (mm, source to sensor; zero = {args.zero_distance}):")
        for i, z in enumerate(distances):
            print(f"  {i:3d}: {z:9.4f}   travel {z - args.zero_distance:9.4f} mm")
        if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
            return 1

    state = RunState()
    cam = Camera(args.gain, args.exposure_us or args.initial_exposure_us)
    cam.start()
    stage = None
    try:
        stage = make_stage(args, required=not args.preview)
        if not args.no_web:
            start_web(cam, state, args.host, args.http_port, stage, args)

        if args.preview:
            state.phase = "preview"
            log.info("Preview mode - Ctrl+C to exit")
            while not state.abort_evt.wait(1.0):
                pass
            return 0

        run_dir = Path(args.out_dir) / dt.datetime.now().strftime("psf_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        fh = logging.FileHandler(run_dir / "capture.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logging.getLogger().addHandler(fh)
        state.run_dir = str(run_dir)
        log.info("Saving to %s", run_dir)

        sweep = PsfSweep(args, cam, stage, state, run_dir)
        try:
            sweep.run(distances)
            if not args.no_home:
                state.phase = "homing"
                log.info("Returning to zero")
                try:
                    stage.zero()
                    state.distance_mm = args.zero_distance
                except ControllerError as e:
                    log.error("Homing failed: %s", e)
            if not args.no_dark and sweep.records and not state.abort_evt.is_set():
                sweep.capture_darks()
        except KeyboardInterrupt:
            sweep.status = "interrupted"
            sweep.write_manifest()
            raise

        v = sweep.validate()
        state.phase = "done"
        print(f"\nRun {sweep.status}: {len(sweep.records)}/{len(distances)} PSFs in {run_dir}")
        if "r2" in v:
            print(f"Scale check (rms radius vs 1/z): R^2={v['r2']:.4f}, effective t={v['t_mm_effective']:.2f} mm, "
                  f"non-monotonic steps={v['radius_increases_with_z_count']}")
            if v["r2"] < 0.95 or v["radius_increases_with_z_count"]:
                print("  ! PSF size doesn't scale cleanly with distance - check the stage moved, "
                      "alignment, and stray light.")
        if state.warnings:
            print(f"{len(state.warnings)} warning(s):")
            for w in state.warnings:
                print("  -", w)
        return 0 if sweep.status == "complete" else 2
    except KeyboardInterrupt:
        log.warning("Interrupted")
        return 130
    finally:
        cam.stop()
        if stage:
            stage.close()
