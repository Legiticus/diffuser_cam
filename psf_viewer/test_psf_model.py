#!/usr/bin/env python3
"""
Checks the PSF model and the run watcher against synthetic runs with a known answer,
plus a leave-one-out check on the newest real run if there is one.

    python3 test_psf_model.py        (or: python3 -m pytest test_psf_model.py)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from psf_model import PsfModel, load_captures, read_manifest
from runs import RunWatcher, find_latest_run
from synthetic import CENTER, SyntheticRun, make_run, render_rate

T_TRUE = 6.0
DISTANCES = [38, 44, 52, 64, 82, 113, 180, 358]
_tmp = tempfile.TemporaryDirectory()
TMP = Path(_tmp.name)
_RUN = make_run(TMP / "ideal", DISTANCES, T_TRUE)
_MODEL: dict = {}
QUIET = dict(log=lambda *a: None)


def model(t_mm=None) -> PsfModel:
    if t_mm not in _MODEL:
        _MODEL[t_mm] = PsfModel.from_run(_RUN, t_mm, **QUIET)
    return _MODEL[t_mm]


def ncc(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# --------------------------------------------------------------------------- model
def test_estimates_t_and_centre_from_run():
    m = model()
    assert m.geometry.ok, m.geometry.note
    assert abs(m.t_mm - T_TRUE) / T_TRUE < 0.02, m.t_mm
    assert np.hypot(m.center[0] - CENTER[0], m.center[1] - CENTER[1]) < 5, m.center


def test_reproduces_capture_at_captured_depth():
    p = model(T_TRUE).predict(0, 0, 64)
    assert [u["distance_mm"] for u in p.info["based_on"]] == [64.0]
    assert p.info["shift_px"] == [0.0, 0.0]
    assert ncc(p.image, render_rate(64, t_mm=T_TRUE)) > 0.99


def test_depth_between_captures():
    for d in (47.0, 70.0, 250.0):
        p = model(T_TRUE).predict(0, 0, d)
        assert len(p.info["based_on"]) == 2 and abs(sum(u["weight"] for u in p.info["based_on"]) - 1) < 1e-9
        gt = render_rate(d, t_mm=T_TRUE)
        assert ncc(p.image, gt) > 0.99, d
        assert abs(p.image.sum() / gt.sum() - 1) < 0.02, d


def test_brightness_follows_the_captures_not_ideal_falloff():
    run = make_run(TMP / "falloff", DISTANCES, T_TRUE, falloff=1.6)
    m = PsfModel.from_run(run, T_TRUE, **QUIET)
    for d in (47.0, 95.0, 250.0):
        gt = render_rate(d, t_mm=T_TRUE, falloff=1.6)
        assert abs(m.predict(0, 0, d).image.sum() / gt.sum() - 1) < 0.03, d


def test_lateral_shift():
    for x, y, d in ((1.5, -1.0, 90.0), (-3.0, 2.0, 45.0)):
        p = model(T_TRUE).predict(x, y, d)
        k = -T_TRUE / (d - T_TRUE) / 0.003
        assert np.allclose(p.info["shift_px"], [k * x, k * y])
        assert ncc(p.image, render_rate(d, x, y, T_TRUE)) > 0.99, (x, y, d)
        assert ncc(p.image, render_rate(d, -x, -y, T_TRUE)) < 0.5      # sign check


def test_extrapolated_depth_uses_end_capture():
    p = model(T_TRUE).predict(0, 0, 500)
    assert p.info["extrapolated_depth"] and len(p.info["based_on"]) == 1
    gt = render_rate(500, t_mm=T_TRUE)
    assert ncc(p.image, gt) > 0.98 and abs(p.image.sum() / gt.sum() - 1) < 0.03


def test_off_sensor_point_flags_unknown_area():
    p = model(T_TRUE).predict(40, -15, 90)
    assert p.info["unknown_area_frac"] > 0.5
    assert p.info["energy_on_sensor_frac"] < 0.2


def test_fov_map():
    frac, x_max, y_max = model(T_TRUE).fov_map(90)
    assert frac[frac.shape[0] // 2, frac.shape[1] // 2] > 0.99
    assert frac[0, 0] < 0.01 and frac[-1, -1] < 0.01
    assert np.isclose(x_max, 1280 * 0.003 * (90 - T_TRUE) / T_TRUE)


def test_rejects_bad_depth_and_t():
    for kw in ({"d_mm": 5}, {"d_mm": 50, "t_mm": -1}, {"d_mm": 50, "t_mm": 40}):
        try:
            model(T_TRUE).predict(0, 0, **kw)
        except ValueError:
            continue
        raise AssertionError(f"should be rejected: {kw}")


# --------------------------------------------------------------------------- watcher
def test_watcher_follows_newest_run_and_live_updates():
    data = TMP / "data"
    w = RunWatcher(data_dir=data, cache_path=TMP / "geo.json", **QUIET)
    assert not w.poll_once() and w.model is None and w.status["phase"] == "waiting"

    old = make_run(data / "psf_20260101_000000", DISTANCES, T_TRUE)
    assert w.poll_once() and w.model.run_dir == old and w.model.geometry.source == "this run"
    assert not w.poll_once()                                      # nothing changed
    v = w.version

    # a new sweep starts: its first PSF alone can't give t, so the last good estimate is used
    live = SyntheticRun(data / "psf_20260102_000000", t_mm=T_TRUE)
    live.add_psf(38.0)
    assert w.poll_once() and w.model.run_dir == live.dir and w.version == v + 1
    assert len(w.model.caps) == 1 and w.model.geometry.source == old.name
    assert not w.model.caps[0].dark_subtracted
    assert w.model.predict(0, 0, 60).image.sum() > 0

    for d in DISTANCES[1:]:                                       # sweep continues
        live.add_psf(float(d))
    v = w.version
    w.poll_once()
    assert w.version == v + 2       # shown at once with the old t, then with the new estimate
    assert len(w.model.caps) == len(DISTANCES) and w.model.manifest["status"] == "running"
    assert w.model.geometry.source == "this run" and abs(w.model.t_mm - T_TRUE) < 0.15

    live.finish()                                                 # dark frames + complete
    w.poll_once()
    assert all(c.dark_subtracted for c in w.model.caps) and w.model.manifest["status"] == "complete"

    # restarting the viewer reuses the cached estimate instead of recomputing
    w2 = RunWatcher(data_dir=data, cache_path=TMP / "geo.json",
                    log=lambda msg: (_ for _ in ()).throw(AssertionError("re-estimated"))
                    if msg.startswith("Estimating") else None)
    w2.poll_once()
    assert w2.model.run_dir == live.dir and abs(w2.model.t_mm - w.model.t_mm) < 1e-9


def test_watcher_reports_when_t_is_unknown():
    data = TMP / "lonely"
    SyntheticRun(data / "psf_20260101_000000", t_mm=T_TRUE).add_psf(40.0)
    w = RunWatcher(data_dir=data, cache_path=None, **QUIET)
    w.poll_once()
    assert w.model is None and w.status["phase"] == "error" and "--t-mm" in w.status["message"]
    w = RunWatcher(data_dir=data, t_mm=T_TRUE, cache_path=None, **QUIET)
    w.poll_once()
    assert w.model is not None and w.model.t_mm == T_TRUE


def test_watcher_fixed_run_and_t_override():
    w = RunWatcher(run_dir=_RUN, t_mm=5.5, cache_path=TMP / "geo.json", **QUIET)
    w.poll_once()
    assert w.model.t_mm == 5.5 and w.model.geometry.source.startswith("given")
    assert np.hypot(w.model.center[0] - CENTER[0], w.model.center[1] - CENTER[1]) < 5


def test_latest_run_ignores_folders_without_manifest():
    d = TMP / "pick"
    make_run(d / "psf_20260101_000000", [38, 60], T_TRUE)
    (d / "psf_20270101_000000").mkdir(parents=True)
    assert find_latest_run(d).name == "psf_20260101_000000"


# --------------------------------------------------------------------------- web
def test_web_api():
    from app import create_app
    empty = RunWatcher(data_dir=TMP / "nothing", cache_path=None, **QUIET)
    empty.poll_once()
    c = create_app(empty).test_client()
    assert c.get("/").status_code == 200
    assert c.get("/api/state").get_json()["phase"] == "waiting"
    assert c.get("/api/predict.json?d=70").status_code == 503

    w = RunWatcher(run_dir=_RUN, t_mm=T_TRUE, cache_path=TMP / "geo.json", **QUIET)
    w.poll_once()
    c = create_app(w).test_client()
    s = c.get("/api/state").get_json()
    assert s["phase"] == "ready" and s["version"] == 1
    info = c.get("/api/info").get_json()
    assert info["depth_range"] == [38.0, 358.0] and len(info["captures"]) == len(DISTANCES)
    assert info["run_status"] == "complete" and info["version"] == 1
    j = c.get("/api/predict.json?x=1&y=2&d=70").get_json()
    assert [u["distance_mm"] for u in j["based_on"]] == [64.0, 82.0]
    assert j["nearest_capture_index"] == 3
    for url in ("/api/predict.png?x=1&y=2&d=70", "/api/fov.png?d=70", "/api/captured/0.png"):
        r = c.get(url)
        assert r.status_code == 200 and r.mimetype == "image/png", url
    r = c.get("/api/predict.npy?x=1&y=2&d=70")
    assert r.status_code == 200 and len(r.data) > 1280 * 800 * 4
    assert c.get("/api/predict.json?d=3").status_code == 400
    assert c.get("/api/predict.json?x=abc").status_code == 400
    assert c.get("/api/predict.json?d=70&t=-2").status_code == 400
    assert c.get("/api/captured/99.png").status_code == 404


# --------------------------------------------------------------------------- real data
def test_real_data_leave_one_out():
    """Predict held-out captures of the newest real run from the others."""
    from app import DEFAULT_DATA
    run = find_latest_run(DEFAULT_DATA)
    if run is None or len(read_manifest(run).get("psfs", [])) < 8:
        print("  (skipped: no real run with >= 8 PSFs)")
        return
    caps = load_captures(run, read_manifest(run))
    full = PsfModel.from_run(run, **QUIET)
    for k in range(2, len(caps) - 1, max(1, len(caps) // 5)):
        held = caps[k]
        m = PsfModel(run, full.manifest, caps[:k] + caps[k + 1:], full.geometry)
        p = m.predict(0, 0, held.distance_mm)
        score = ncc(p.image, held.image)
        energy_err = p.image.sum() / held.energy - 1
        print(f"  {run.name} d={held.distance_mm:6.1f}: ncc {score:.3f}, energy {energy_err:+.1%}")
        assert score > 0.93, (held.distance_mm, score)
        assert abs(energy_err) < 0.2, (held.distance_mm, energy_err)


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
