"""Full capture flow with the fake camera and fake stage."""

import time

import pytest

from psf_capture import storage
from psf_capture.core import Busy
from psf_capture.session import exposure_ladder

from conftest import run_session, wait_for

SMALL = {"x_mm": 2.0, "y_mm": -1.5, "z_min": 20, "z_max": 200, "n_positions": 3,
         "frames_per_position": 2, "passes": 2, "settle_s": 0}


def test_full_session_writes_frames_metadata_and_backgrounds(backend):
    sid = run_session(backend, SMALL)
    d = backend.data_dir / sid
    assert sid.endswith("_x+2.0_y-1.5")
    m = storage.load_metadata(d)
    assert m["status"] == "complete"
    assert m["plan"]["actual_z"][0] == 20 and m["plan"]["actual_z"][-1] == 200
    assert len(m["frames"]) == 3 * 2 * 2
    assert (d / "z002_p2_f01.tiff").exists()
    raw = storage.read_frame(d / "z000_p1_f00.tiff")
    assert raw.dtype.name == "uint16" and raw.shape == (800, 1280) and raw.max() <= 1023
    # per-position exposure keeps the brightest pixel below saturation, near the 75% target
    for f in m["frames"]:
        assert f["saturated_px"] == 0
        assert 0.3 < (f["peak"] - 64) / (1023 - 64) < 0.76
    # exposure falls off with distance => longer exposures further away
    exps = [p["requested_exposure_us"] for p in m["positions"] if p["pass"] == 1]
    assert exps == sorted(exps) and exps[0] < exps[-1]
    # a background at every exposure used, taken with the light off
    used = {f["requested_exposure_us"] for f in m["frames"]}
    assert {b["requested_exposure_us"] for b in m["backgrounds"]} == used
    bg = storage.read_frame(d / m["backgrounds"][0]["file"])
    assert bg.mean() < 80
    assert not m["homing_check"]["steps_lost"]
    assert m["homing_check"]["offset_mm"] == 0 and m["homing_baseline_mm"] > 0
    assert m["app"]["version"]


def test_fixed_exposure_warns_on_saturation(backend):
    backend.set_live_controls(200000, 1.0)       # far too long near the diffuser
    sid = run_session(backend, {**SMALL, "exposure_mode": "fixed", "passes": 1, "background": False})
    m = storage.load_metadata(backend.data_dir / sid)
    assert {f["requested_exposure_us"] for f in m["frames"]} == {200000}
    assert any("saturated" in w for w in m["warnings"])
    assert m["backgrounds"] == []


def test_no_psf_when_signal_too_weak(backend):
    backend.world.brightness = 1e-6
    sid = run_session(backend, {**SMALL, "passes": 1, "n_positions": 1, "max_exposure_us": 5000})
    m = storage.load_metadata(backend.data_dir / sid)
    assert m["positions"][0]["no_psf"] and m["frames"] == []
    assert any("No PSF" in w for w in m["warnings"])


def test_manual_commands_locked_while_capturing_and_abort(backend):
    backend.start_session({**SMALL, "n_positions": 30, "settle_s": 0.2})
    wait_for(lambda: backend.progress.get("phase") in ("moving", "exposing", "capturing"))
    with pytest.raises(Busy):
        backend.move(50)
    with pytest.raises(Busy):
        backend.set_live_controls(1000, 1)
    with pytest.raises(Busy):
        backend.start_session(SMALL)
    backend.abort()
    wait_for(lambda: backend.state == "idle")
    m = storage.load_metadata(backend.data_dir / backend.session_id)
    assert m["status"] == "aborted"
    for f in m["frames"]:                          # what was taken before the abort is kept
        assert (backend.data_dir / backend.session_id / f["file"]).exists()


def test_pause_blocks_progress(backend):
    backend.start_session({**SMALL, "n_positions": 30})
    wait_for(lambda: backend.progress.get("frames", 0) > 0)
    backend.pause(True)
    time.sleep(0.5)
    frames = backend.progress["frames"]
    time.sleep(0.5)
    assert backend.progress["frames"] == frames
    backend.pause(False)
    wait_for(lambda: backend.progress["frames"] > frames)
    backend.abort()
    wait_for(lambda: backend.state == "idle")


def test_requires_xy_and_valid_range(backend):
    with pytest.raises(ValueError):
        backend.start_session({"z_min": 10, "z_max": 340})
    with pytest.raises(ValueError):
        backend.start_session({**SMALL, "z_max": 400})
    assert backend.state == "idle"


def test_exposure_ladder():
    ladder = exposure_ladder(9, 100000)
    assert ladder[0] == 9 and ladder[-1] == 100000
    ratios = [b / a for a, b in zip(ladder[1:], ladder[2:-1])]
    assert all(1.3 < r < 1.5 for r in ratios)
