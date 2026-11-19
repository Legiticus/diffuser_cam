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
    stats = {"background": bg, "noise": noise, "peak": peak, "snr_peak": (peak - bg) / noise,
             "rms_radius_px": None, "edge_energy_frac": None}
    if total > 0:
        ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        cx = (w * xs).sum() / total
        cy = (w * ys).sum() / total
        stats["rms_radius_px"] = math.sqrt(float((w * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / total))
        edge = w.copy()
        edge[edge_px:-edge_px, edge_px:-edge_px] = 0
        stats["edge_energy_frac"] = float(edge.sum() / total)
    return stats


def to_display(raw: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """10-bit raw -> contrast-stretched 8-bit BGR. Saturated pixels drawn red."""
    img = raw.astype(np.float32)
    lo = float(np.percentile(img[::4, ::4], 1))
    hi = max(float(np.percentile(img[::4, ::4], 99.9)), lo + 8)
    g = np.clip((img - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    bgr[raw >= SENSOR_MAX] = (0, 0, 255)
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr
