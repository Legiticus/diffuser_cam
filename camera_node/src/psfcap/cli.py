"""Command-line interface and program entry point."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

from .camera import Camera
from .constants import (DEFAULT_OUT_DIR, HTTP_PORT, INITIAL_EXPOSURE_US, MAX_TRAVEL_MM,
                        ZERO_DISTANCE_MM)
from .controller import ControllerError, StageController, find_serial_port
from .state import RunState
from .sweep import PsfSweep, plan_distances
from .web import start_web

log = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DiffuserCam PSF depth-sweep capture",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stop", type=float, required=True,
                   help=f"last distance (mm), at most {ZERO_DISTANCE_MM + MAX_TRAVEL_MM:.0f}")
    p.add_argument("--start", type=float, default=ZERO_DISTANCE_MM, help="first distance (mm)")
    p.add_argument("--num", type=int, default=20, help="number of distances (uniform in 1/z)")
    p.add_argument("--frames", type=int, default=10, help="frames averaged per PSF")
    p.add_argument("--port", default="auto", help="controller serial port, or 'auto'")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("picamera2", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    distances = plan_distances(args.start, args.stop, args.num)
    print(f"\nPlanned {len(distances)} PSF distances (mm, source to sensor; zero = {ZERO_DISTANCE_MM}):")
    for i, z in enumerate(distances):
        print(f"  {i:3d}: {z:9.4f}   travel {z - ZERO_DISTANCE_MM:9.4f} mm")

    state = RunState()
    cam = Camera(INITIAL_EXPOSURE_US)
    cam.start()
    stage = None
    try:
        stage = StageController(find_serial_port() if args.port == "auto" else args.port)
        stage.open()
        start_web(cam, state, HTTP_PORT)

        run_dir = Path(args.out_dir) / dt.datetime.now().strftime("psf_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        state.run_dir = str(run_dir)
        log.info("Saving to %s", run_dir)

        sweep = PsfSweep(cam, stage, state, run_dir, args.frames)
        sweep.run(distances)
        state.phase = "homing"
        log.info("Returning to zero")
        try:
            stage.zero()
            state.distance_mm = ZERO_DISTANCE_MM
        except ControllerError as e:
            log.error("Homing failed: %s", e)
        sweep.capture_darks()

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
