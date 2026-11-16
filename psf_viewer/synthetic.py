"""
Synthetic PSF capture runs with a known diffuser-to-sensor distance t and scale
centre, in the format psf_capture writes (manifest.json, psf_*.npy, dark_*.npy).
Used by the tests, and to try the viewer without real data:

    python3 synthetic.py /tmp/demo/psf_20260101_000000 --t 2.6
    python3 app.py --data-dir /tmp/demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 800
PITCH_MM = 0.003
DARK_LEVEL = 64.0
CENTER = (760.0, 430.0)        # scale centre, deliberately away from the image centre


def _features(seed: int, n: int = 1500, radius_px: float = 300.0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (n * 2, 2))
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < 1][:n] * radius_px
    return pts, rng.uniform(0.3, 1.0, len(pts))


def render_rate(d_mm: float, x_mm: float = 0.0, y_mm: float = 0.0, t_mm: float = 6.0,
                seed: int = 0, center=CENTER, falloff: float = 2.0) -> np.ndarray:
    """Noise-free sensor image (counts per microsecond) for a point at (x, y, d).
    Total light ~ 1/z^falloff (2 is ideal)."""
    pts, amp = _features(seed)
    z = d_mm - t_mm
    m = d_mm / z
    shift = -t_mm / z * np.array([x_mm, y_mm]) / PITCH_MM
    pos = np.array(center) + m * pts + shift
    img = np.zeros((H, W), np.float64)
    x0 = np.floor(pos[:, 0]).astype(int)
    y0 = np.floor(pos[:, 1]).astype(int)
    fx, fy = pos[:, 0] - x0, pos[:, 1] - y0
    a = amp * (32.0 / z) ** falloff
    for ox, oy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + ox, y0 + oy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        np.add.at(img, (yi[ok], xi[ok]), a[ok] * wgt[ok])
    img = cv2.GaussianBlur(img, (0, 0), 1.5 * m)
    return img.astype(np.float32)


class SyntheticRun:
    """Writes a run step by step like psf_capture does (manifest rewritten after
    every PSF, dark frames at the end), so the viewer's live updates can be tested."""

    def __init__(self, out_dir: Path, t_mm: float = 6.0, seed: int = 0, noise: float = 1.5,
                 falloff: float = 2.0):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t, self.seed, self.noise, self.falloff = t_mm, seed, noise, falloff
        self.rng = np.random.default_rng(seed + 1)
        self.recs: list[dict] = []
        self.status = "running"
        self.write_manifest()

    def write_manifest(self):
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps({
            "created": "synthetic", "sensor": "synthetic", "synthetic_t_mm": self.t,
            "synthetic_center_px": list(CENTER), "zero_distance_mm": 38.0, "raw_size_wh": [W, H],
            "status": self.status, "warnings": [], "psfs": self.recs}, indent=2))
        tmp.replace(self.dir / "manifest.json")

    def add_psf(self, d: float):
        rate = render_rate(d, t_mm=self.t, seed=self.seed, falloff=self.falloff)
        exp = int(0.75 * (1023 - DARK_LEVEL) / max(rate.max(), 1e-9))
        counts = np.clip(DARK_LEVEL + rate * exp + self.rng.normal(0, self.noise, rate.shape), 0, 1023)
        i = len(self.recs)
        stem = f"psf_{i:03d}_z{d:08.3f}mm"
        np.save(self.dir / f"{stem}.npy", counts.astype(np.float32))
        self.recs.append({"index": i, "distance_mm": d, "file": f"{stem}.npy", "exposure_us": exp,
                          "requested_exposure_us": exp, "n_frames": 10, "warnings": []})
        self.write_manifest()

    def add_darks(self):
        for r in self.recs:
            name = f"dark_exp{int(r['exposure_us']):08d}us.npy"
            if not (self.dir / name).exists():
                np.save(self.dir / name, (DARK_LEVEL + self.rng.normal(0, self.noise / 3, (H, W))).astype(np.float32))
            r["dark_file"] = name
        self.write_manifest()

    def finish(self, darks: bool = True):
        if darks:
            self.add_darks()
        self.status = "complete"
        self.write_manifest()


def make_run(out_dir: Path, distances, t_mm: float = 6.0, seed: int = 0, noise: float = 1.5,
             falloff: float = 2.0) -> Path:
    run = SyntheticRun(out_dir, t_mm, seed, noise, falloff)
    for d in distances:
        run.add_psf(float(d))
    run.finish()
    return Path(out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--t", type=float, default=6.0, help="diffuser-to-sensor distance (mm)")
    p.add_argument("--distances", default="38,44,52,64,82,113,180,358")
    a = p.parse_args()
    make_run(Path(a.out_dir), [float(x) for x in a.distances.split(",")], a.t)
    print(f"wrote synthetic run to {a.out_dir}")
