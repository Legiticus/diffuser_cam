"""PSF generator built from processed sessions.

In-between PSFs are never made by blending neighbouring measurements pixel by pixel
(two shifted caustics blend into two faint copies). Instead:
  - lateral: take the measured PSF at the nearest (x, y) offset and shift it by
    k(z) * (dx, dy) px, where k(z) is the shift per mm fitted from sessions at
    different offsets (or d / (z * pitch) with d = 2.3 mm if only one offset exists)
  - depth: use the nearest measured z plane of that session
Several points are summed (valid for an incoherent source). Points outside the
measured x, y or z range produce warnings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import processing
from .config import SHIFT_SIGN_COL, SHIFT_SIGN_ROW


class PsfModel:
    def __init__(self, cache_dirs: list[Path], progress=None):
        if not cache_dirs:
            raise ValueError("Select at least one session")
        self.sessions = []                        # (psf stack (mmap), info)
        for d in cache_dirs:
            self.sessions.append(processing.load(d))
        self.warnings: list[str] = []
        plans = {tuple(info["z_mm"]) for _, info in self.sessions}
        if len(plans) > 1:
            names = sorted({str(info["plan"].get("name")) for _, info in self.sessions})
            self.warnings.append("The selected sessions used different z sampling plans "
                                 f"({', '.join(names)}); their z planes do not line up")
        units = {info["steps"]["normalize"] for _, info in self.sessions}
        if len(units) > 1:
            self.warnings.append("The selected sessions were normalised differently")
        self.xs = [info["x_mm"] for _, info in self.sessions]
        self.ys = [info["y_mm"] for _, info in self.sessions]
        self.k_fit = self._fit_shift_per_mm(progress)

    # --- fitting ------------------------------------------------------------------------
    def _fit_shift_per_mm(self, progress) -> dict:
        """Least-squares shift per mm at each z of the reference session, from how the
        PSF moves between sessions at different offsets. {z_mm: px per mm}."""
        ref_i = int(np.argmin(np.hypot(self.xs, self.ys)))       # session closest to the axis
        ref_psf, ref_info = self.sessions[ref_i]
        others = [s for k, s in enumerate(self.sessions) if k != ref_i
                  and (s[1]["x_mm"], s[1]["y_mm"]) != (ref_info["x_mm"], ref_info["y_mm"])]
        fit = {}
        if not others:
            return fit
        for zi, z in enumerate(ref_info["z_mm"]):
            if progress:
                progress(zi / len(ref_info["z_mm"]), f"fitting shift per mm: z {zi + 1}")
            if ref_info["no_psf"][zi]:
                continue
            num = den = 0.0
            for psf, info in others:
                zj = int(np.argmin(np.abs(np.array(info["z_mm"]) - z)))
                if info["no_psf"][zj] or abs(info["z_mm"][zj] - z) > 1e-6 + 0.01 * z:
                    continue
                dy, dx, quality = processing.phase_correlate(processing._bin2(ref_psf[zi]),
                                                             processing._bin2(psf[zj]))
                if quality < 0.05:
                    continue
                u_row = SHIFT_SIGN_ROW * (info["y_mm"] - ref_info["y_mm"])
                u_col = SHIFT_SIGN_COL * (info["x_mm"] - ref_info["x_mm"])
                num += 2 * dy * u_row + 2 * dx * u_col
                den += u_row ** 2 + u_col ** 2
            if den > 0:
                fit[z] = num / den
        if fit and np.median(list(fit.values())) < 0:
            self.warnings.append("The fitted shift per mm is negative: the image axes are "
                                 "flipped relative to SHIFT_SIGN_* in config.py")
        return fit

    def shift_per_mm(self, z: float) -> float:
        if not self.k_fit:
            return processing.shift_per_mm(z)
        zs = np.array(sorted(self.k_fit))
        ks = np.array([self.k_fit[v] for v in zs])
        # k ~ c / z, so interpolate k * z and divide by z again
        return float(np.interp(z, zs, ks * zs) / z)

    def summary(self) -> dict:
        fitted = [{"z_mm": z, "px_per_mm": k, "d_mm": k * z * processing.PIXEL_PITCH_MM}
                  for z, k in sorted(self.k_fit.items())]
        return {
            "sessions": [info["session"] for _, info in self.sessions],
            "x_range": [min(self.xs), max(self.xs)], "y_range": [min(self.ys), max(self.ys)],
            "z_range": [min(min(i["z_mm"]) for _, i in self.sessions),
                        max(max(i["z_mm"]) for _, i in self.sessions)],
            "shift_fit": fitted,
            "fitted_d_mm": float(np.median([f["d_mm"] for f in fitted])) if fitted else None,
            "warnings": self.warnings,
        }

    # --- generation ---------------------------------------------------------------------
    def generate(self, points: list[dict]) -> tuple[np.ndarray, list[str], list[dict]]:
        """Sum of the predicted PSFs. Returns (image float32, warnings, per-point details)."""
        if not points:
            raise ValueError("No points given")
        warnings, details = [], []
        h, w = self.sessions[0][0].shape[1:]
        total = np.zeros((h, w), np.float32)
        tol = 1e-6
        for n, p in enumerate(points, 1):
            x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
            intensity = float(p.get("intensity", 1.0))
            if not min(self.xs) - tol <= x <= max(self.xs) + tol:
                warnings.append(f"Point {n}: x={x} mm is outside the measured x range "
                                f"{min(self.xs)}..{max(self.xs)} mm")
            if not min(self.ys) - tol <= y <= max(self.ys) + tol:
                warnings.append(f"Point {n}: y={y} mm is outside the measured y range "
                                f"{min(self.ys)}..{max(self.ys)} mm")
            s = int(np.argmin(np.hypot(np.array(self.xs) - x, np.array(self.ys) - y)))
            psf, info = self.sessions[s]
            zs = np.array(info["z_mm"])
            if not zs.min() - tol <= z <= zs.max() + tol:
                warnings.append(f"Point {n}: z={z} mm is outside the measured z range "
                                f"{zs.min():.2f}..{zs.max():.2f} mm")
            order = np.argsort(np.abs(zs - z))
            usable = [i for i in order if not info["no_psf"][i]]
            if not usable:
                raise ValueError(f"Session {info['session']} has no usable PSF planes")
            zi = int(usable[0])
            if zi != order[0]:
                warnings.append(f"Point {n}: nearest plane z={zs[order[0]]:.2f} mm has no PSF; "
                                f"used z={zs[zi]:.2f} mm")
            k = self.shift_per_mm(zs[zi])
            dy = SHIFT_SIGN_ROW * k * (y - info["y_mm"])
            dx = SHIFT_SIGN_COL * k * (x - info["x_mm"])
            plane = np.asarray(psf[zi], np.float32)
            shifted = processing.shift_image(plane, dy, dx) if (dy or dx) else plane
            total += intensity * np.nan_to_num(shifted)
            details.append({"point": n, "session": info["session"], "z_plane_mm": float(zs[zi]),
                            "shift_px": [dy, dx], "invalid_px": int(np.isnan(shifted).sum())})
        return total, warnings, details
