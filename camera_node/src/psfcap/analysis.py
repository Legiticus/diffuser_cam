"""PSF statistics and display conversion."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .constants import SENSOR_MAX


def psf_stats(img: np.ndarray, edge_px: int = 8) -> dict:
    """Statistics used to judge a PSF. img: float array in raw DN."""
    small = img[::4, ::4]
    bg = float(np.percentile(small, 1))
    low = small[small <= np.percentile(small, 50)]
    noise = float(1.4826 * np.median(np.abs(low - np.median(low)))) + 1e-6
    peak = float(np.percentile(img, 99.995))
    w = np.clip(img - bg - 3 * noise, 0, None)
    total = float(w.sum())
    h, wd = img.shape
    stats = {"background": bg, "noise": noise, "peak": peak, "max": float(img.max()),
             "mean": float(img.mean()), "signal_sum": total,
             "snr_peak": (peak - bg) / noise}
    if total > 0:
        ys, xs = np.mgrid[0:h, 0:wd]
        cx = float((w * xs).sum() / total)
        cy = float((w * ys).sum() / total)
        r2 = float((w * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / total)
        edge = w.copy()
        edge[edge_px:-edge_px, edge_px:-edge_px] = 0
        stats |= {"centroid_x": cx, "centroid_y": cy, "rms_radius_px": math.sqrt(r2),
                  "edge_energy_frac": float(edge.sum() / total)}
    else:
        stats |= {"centroid_x": None, "centroid_y": None, "rms_radius_px": None,
                  "edge_energy_frac": None}
    return stats


def to_display(raw: np.ndarray, stretch: bool, scale: float = 1.0,
               sat_level: int = SENSOR_MAX) -> np.ndarray:
    """10-bit raw -> 8-bit BGR. Saturated pixels drawn red."""
    img = raw.astype(np.float32)
    if stretch:
        lo = float(np.percentile(img[::4, ::4], 1))
        hi = max(float(np.percentile(img[::4, ::4], 99.9)), lo + 8)
    else:
        lo, hi = 0.0, float(SENSOR_MAX)
    g = np.clip((img - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    sat = raw >= sat_level
    if sat.any():
        bgr[sat] = (0, 0, 255)
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr
