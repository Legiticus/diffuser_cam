#!/usr/bin/env python3
"""
Checks the PSF model against a synthetic run with a known answer.

    python3 test_psf_model.py        (or: python3 -m pytest test_psf_model.py)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from psf_model import PsfModel
from synthetic import make_run, render_rate

T_TRUE = 6.0
DISTANCES = [38, 44, 52, 64, 82, 113, 180, 358]
_tmp = tempfile.TemporaryDirectory()
_RUN = make_run(Path(_tmp.name) / "run", DISTANCES, T_TRUE)
_MODEL: dict = {}


def model(t_mm=None) -> PsfModel:
    if t_mm not in _MODEL:
        _MODEL[t_mm] = PsfModel(_RUN, t_mm, log=lambda *a: None)
    return _MODEL[t_mm]


def ncc(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def test_estimates_t_from_run():
    m = model()
    assert m.t_estimate.note == "ok", m.t_estimate.note
    assert abs(m.t_mm - T_TRUE) / T_TRUE < 0.02, m.t_mm


def test_reproduces_capture_at_captured_depth():
    p = model(T_TRUE).predict(0, 0, 64)
    assert p.info["scale_from_capture"] == 1.0 and p.info["shift_px"] == [0.0, 0.0]
    assert ncc(p.image, render_rate(64, t_mm=T_TRUE)) > 0.99


def test_depth_between_captures():
    for d in (47.0, 70.0, 250.0):
        p = model(T_TRUE).predict(0, 0, d)
        gt = render_rate(d, t_mm=T_TRUE)
        assert ncc(p.image, gt) > 0.99, d
        assert abs(p.image.sum() / gt.sum() - 1) < 0.02, d       # brightness falls off correctly


def test_lateral_shift():
    for x, y, d in ((1.5, -1.0, 90.0), (-3.0, 2.0, 45.0)):
        p = model(T_TRUE).predict(x, y, d)
        expected = -T_TRUE / (d - T_TRUE) / 0.003
        assert np.allclose(p.info["shift_px"], [expected * x, expected * y])
        assert ncc(p.image, render_rate(d, x, y, T_TRUE)) > 0.99, (x, y, d)
        wrong_way = render_rate(d, -x, -y, T_TRUE)                # sign check
        assert ncc(p.image, wrong_way) < 0.5


def test_off_sensor_point_flags_unknown_area():
    p = model(T_TRUE).predict(40, -15, 90)
    assert p.info["unknown_area_frac"] > 0.5
    assert p.info["captured_energy_on_sensor"] < 0.2


def test_fov_map():
    frac, x_max, y_max = model(T_TRUE).fov_map(90)
    assert frac[frac.shape[0] // 2, frac.shape[1] // 2] > 0.99
    assert frac[0, 0] < 0.01 and frac[-1, -1] < 0.01
    assert np.isclose(x_max, 1280 * 0.003 * (90 - T_TRUE) / T_TRUE)


def test_rejects_depth_inside_diffuser_gap():
    try:
        model(T_TRUE).predict(0, 0, 5)
    except ValueError:
        return
    raise AssertionError("depth <= t should be rejected")


def test_web_api():
    from app import create_app
    c = create_app(model(T_TRUE)).test_client()
    assert c.get("/").status_code == 200
    info = c.get("/api/info").get_json()
    assert info["depth_range"] == [38.0, 358.0] and len(info["captures"]) == len(DISTANCES)
    j = c.get("/api/predict.json?x=1&y=2&d=70").get_json()
    assert j["nearest_capture"]["distance_mm"] == 64.0
    for url in ("/api/predict.png?x=1&y=2&d=70", "/api/fov.png?d=70", "/api/captured/0.png"):
        r = c.get(url)
        assert r.status_code == 200 and r.mimetype == "image/png", url
    r = c.get("/api/predict.npy?x=1&y=2&d=70")
    assert r.status_code == 200 and len(r.data) > 1280 * 800 * 4
    assert c.get("/api/predict.json?d=3").status_code == 400
    assert c.get("/api/predict.json?x=abc").status_code == 400


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
