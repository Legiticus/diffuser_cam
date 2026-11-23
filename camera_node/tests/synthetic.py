"""Build session folders from known synthetic PSFs (no capture loop), for processing tests."""

from datetime import datetime

import numpy as np

from psf_capture import plans, storage
from psf_capture.camera import FakeCamera, FakeWorld

BLACK = 64
_cam = None


def camera():
    global _cam
    if _cam is None:
        _cam = FakeCamera(FakeWorld())
    return _cam


def make_session(data_dir, x=0.0, y=0.0, z=(30.0, 60.0, 120.0), passes=1, frames=2,
                 exposure=1000, pass_shift=None, ambient=5.0, source_x=None, blur_pass=None,
                 plan_name="test", noise=0.0, seed=0):
    """Write a complete session. Signal = rendered PSF (counts per us at gain 1) * exposure.
    pass_shift(p, i) -> (dy, dx) extra jitter; source_x(z) overrides the true x (rail tilt)."""
    rng = np.random.default_rng(seed)
    plan = plans.positions({"name": plan_name, "z_min": z[0], "z_max": z[-1],
                            "n_positions": len(z), "spacing": "linear"})
    plan["actual_z"] = list(z)
    plan.pop("warnings")
    sid = storage.session_name(datetime(2026, 1, 1, 0, 0, int(seed) % 60), x, y)
    d = data_dir / sid
    d.mkdir(parents=True)
    meta = {"status": "complete", "x_mm": x, "y_mm": y, "plan": plan, "ended": "t",
            "options": {"passes": passes}, "frames": [], "backgrounds": [], "positions": [],
            "camera": {"width": 1280, "height": 800, "black_level": BLACK, "saturation_level": 1023}}
    truth = {}
    for i, zz in enumerate(z):
        sx = source_x(zz) if source_x else x
        for p in range(1, passes + 1):
            jitter = pass_shift(p, i) if pass_shift else (0.0, 0.0)
            signal = camera().render(sx, y, zz, jitter) * exposure
            if blur_pass == p:
                signal = 0.25 * (signal + np.roll(signal, 3, 1) + np.roll(signal, 6, 1) + np.roll(signal, 9, 1))
            truth[(i, p)] = signal
            for f in range(frames):
                raw = signal + BLACK + ambient + rng.normal(0, noise, signal.shape)
                name = storage.frame_name(i, p, f)
                storage.write_frame(d / name, np.clip(np.rint(raw), 0, 1023).astype(np.uint16))
                meta["frames"].append({"file": name, "z_index": i, "pass": p, "frame": f, "z_mm": zz,
                                       "requested_exposure_us": exposure, "exposure_us": exposure,
                                       "gain": 1.0})
    for f in range(frames):
        name = storage.background_name(exposure, f)
        storage.write_frame(d / name, np.full((800, 1280), BLACK + ambient, np.uint16))
        meta["backgrounds"].append({"file": name, "requested_exposure_us": exposure, "frame": f,
                                    "exposure_us": exposure, "gain": 1.0})
    storage.write_json(d / storage.METADATA, meta)
    return d, truth
