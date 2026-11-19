"""
Synthetic PSF capture run with a known diffuser-to-sensor distance t, in the same
format psf_capture writes (manifest.json, psf_*.npy, dark_*.npy). Used by the
tests, and to try the viewer without real data:

    python3 synthetic.py /tmp/demo_run --t 6
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


def _features(seed: int, n: int = 1500, radius_px: float = 300.0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (n * 2, 2))
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < 1][:n] * radius_px
    return pts, rng.uniform(0.3, 1.0, len(pts))


def render_rate(d_mm: float, x_mm: float = 0.0, y_mm: float = 0.0, t_mm: float = 6.0,
                seed: int = 0, center=(660.0, 390.0)) -> np.ndarray:
    """Noise-free sensor image (counts per microsecond) for a point at (x, y, d)."""
    pts, amp = _features(seed)
    z = d_mm - t_mm
    m = d_mm / z
    shift = -t_mm / z * np.array([x_mm, y_mm]) / PITCH_MM
    pos = np.array(center) + m * pts + shift
    img = np.zeros((H, W), np.float64)
    x0 = np.floor(pos[:, 0]).astype(int)
    y0 = np.floor(pos[:, 1]).astype(int)
    fx, fy = pos[:, 0] - x0, pos[:, 1] - y0
    a = amp * (32.0 / z) ** 2            # collected light ~ 1/z^2
    for ox, oy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + ox, y0 + oy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        np.add.at(img, (yi[ok], xi[ok]), a[ok] * wgt[ok])
    img = cv2.GaussianBlur(img, (0, 0), 1.5 * m)
    return img.astype(np.float32)


def make_run(out_dir: Path, distances, t_mm: float = 6.0, seed: int = 0, noise: float = 1.5) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 1)
    recs = []
    for i, d in enumerate(distances):
        rate = render_rate(d, t_mm=t_mm, seed=seed)
        exp = int(0.75 * (1023 - DARK_LEVEL) / max(rate.max(), 1e-9))
        counts = np.clip(DARK_LEVEL + rate * exp + rng.normal(0, noise, rate.shape), 0, 1023)
        stem = f"psf_{i:03d}_z{d:08.3f}mm"
        dark = f"dark_exp{exp:08d}us.npy"
        np.save(out_dir / f"{stem}.npy", counts.astype(np.float32))
        np.save(out_dir / dark, (DARK_LEVEL + rng.normal(0, noise / 3, rate.shape)).astype(np.float32))
        recs.append({"index": i, "distance_mm": d, "file": f"{stem}.npy", "exposure_us": exp,
                     "requested_exposure_us": exp, "n_frames": 10, "dark_file": dark, "warnings": []})
    (out_dir / "manifest.json").write_text(json.dumps({
        "sensor": "synthetic", "synthetic_t_mm": t_mm, "zero_distance_mm": 38.0,
        "raw_size_wh": [W, H], "status": "complete", "psfs": recs}, indent=2))
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--t", type=float, default=6.0, help="diffuser-to-sensor distance (mm)")
    p.add_argument("--distances", default="38,44,52,64,82,113,180,358")
    a = p.parse_args()
    make_run(Path(a.out_dir), [float(x) for x in a.distances.split(",")], a.t)
    print(f"wrote synthetic run to {a.out_dir}")
