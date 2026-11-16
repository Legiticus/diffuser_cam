"""
Forward model that predicts the sensor PSF for any point, from a PSF capture run.

DiffuserCam model (Antipa et al., "DiffuserCam: lensless single-exposure 3D imaging"):

  * A diffuser sits a distance t in front of the sensor. A point source at
    source-to-sensor distance d is z = d - t from the diffuser.
  * Lateral shift invariance: moving the point sideways by (x, y) translates the
    caustic pattern on the sensor by  -(t / z) * (x, y).
  * Depth scaling: the caustic pattern is magnified by  M(d) = d / (d - t)
    (1 + t/z) about the point's projection on the sensor. The light collected
    falls off as 1/z^2.

Prediction for a point (x, y, d):
  1. take the captured PSF whose depth is closest in 1/d,
  2. dark-subtract it and divide by exposure (counts per microsecond),
  3. scale it by M(d) / M(d_i) about its centroid (energy conserving, times the
     (z_i / z)^2 brightness change),
  4. translate it by -(t / z) * (x, y) / pixel_pitch.

Regions of the prediction that come from outside the captured sensor area are
unknown and are reported as such (mask), never invented.

t is estimated from the run itself: every PSF is compared to the nearest one to
measure their relative scale (FFT cross-correlation over a range of scales, which
is insensitive to translation), and t is fitted to the scale-vs-depth model. If
you know t from the build, pass it explicitly; it's more accurate.

Object coordinates (x, y) are in mm, measured along the sensor's column (x) and
row (y) directions from the axis the light source travelled along.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PIXEL_PITCH_UM = 3.0       # OV9281
SENSOR_MAX = 1023


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class CapturedPsf:
    index: int
    distance_mm: float
    file: str
    exposure_us: float
    image: np.ndarray          # float32, dark-subtracted, counts per microsecond (unclipped:
                               # clipping noise at 0 would bias the energy upward)
    centroid: tuple[float, float]
    dark_subtracted: bool
    warnings: list = field(default_factory=list)


def load_run(run_dir: Path) -> tuple[dict, list[CapturedPsf]]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    psfs = []
    for rec in manifest["psfs"]:
        raw = np.load(run_dir / rec["file"]).astype(np.float32)
        dark_file = rec.get("dark_file")
        if dark_file and (run_dir / dark_file).exists():
            img = raw - np.load(run_dir / dark_file).astype(np.float32)
            dark = True
        else:   # no dark frame: estimate a flat background
            img = raw - float(np.percentile(raw[::4, ::4], 1))
            dark = False
        img = img / float(rec["exposure_us"])
        psfs.append(CapturedPsf(rec["index"], float(rec["distance_mm"]), rec["file"],
                                float(rec["exposure_us"]), img.astype(np.float32),
                                centroid(img), dark, list(rec.get("warnings", []))))
    if not psfs:
        raise ValueError(f"No PSFs in {run_dir / 'manifest.json'}")
    psfs.sort(key=lambda p: p.distance_mm)
    return manifest, psfs


def centroid(img: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(np.clip(img, 0, None).astype(np.float32))
    if m["m00"] <= 0:
        return (img.shape[1] / 2, img.shape[0] / 2)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def magnification(d_mm: float, t_mm: float) -> float:
    return d_mm / (d_mm - t_mm)


def _scale_matrix(s: float, cx: float, cy: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Scale by s about (cx, cy), then translate by (dx, dy)."""
    return np.float32([[s, 0, (1 - s) * cx + dx], [0, s, (1 - s) * cy + dy]])


# --------------------------------------------------------------------------- #
# Estimating t (diffuser-to-sensor distance) from the run
# --------------------------------------------------------------------------- #
def _normalise(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def _corr_peak(target_f: np.ndarray, moving: np.ndarray) -> float:
    """Max of the normalised circular cross-correlation (any translation)."""
    m = np.fft.rfft2(_normalise(moving))
    return float(np.fft.irfft2(target_f * np.conj(m), s=moving.shape).max())


def relative_scale(ref: np.ndarray, target: np.ndarray, lo: float = 0.4, hi: float = 1.05
                   ) -> tuple[float, float]:
    """Scale s such that ref scaled by s best matches target (up to translation).
    Returns (s, correlation peak)."""
    h, w = ref.shape

    def score(r, t_f, s):
        rh, rw = r.shape
        warped = cv2.warpAffine(r, _scale_matrix(s, rw / 2, rh / 2), (rw, rh), flags=cv2.INTER_LINEAR)
        return _corr_peak(t_f, warped)

    def search(r, t, grid):
        t_f = np.fft.rfft2(_normalise(t))
        sc = np.array([score(r, t_f, s) for s in grid])
        return grid, sc

    def down(img, f):
        return cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)

    # coarse-to-fine: quarter res over the full range, half res, then full res
    g, sc = search(down(ref, 4), down(target, 4), np.arange(lo, hi, 0.004))
    s0 = float(g[int(np.argmax(sc))])
    g, sc = search(down(ref, 2), down(target, 2), np.arange(s0 - 0.006, s0 + 0.00601, 0.001))
    s0 = float(g[int(np.argmax(sc))])
    fine, fs = search(ref, target, np.arange(s0 - 0.0015, s0 + 0.00151, 0.0005))
    k = int(np.argmax(fs))
    s_best = float(fine[k])
    if 0 < k < len(fine) - 1:
        a, b, c = fs[k - 1], fs[k], fs[k + 1]
        den = a - 2 * b + c
        if den < 0:
            s_best += 0.5 * (a - c) / den * 0.0005
    return s_best, float(fs[k])


@dataclass
class TEstimate:
    t_mm: float | None
    rms_residual: float | None
    scales: list           # [(distance_mm, measured s, correlation)]
    note: str


def estimate_t(psfs: list[CapturedPsf], log=print) -> TEstimate:
    if len(psfs) < 2:
        return TEstimate(None, None, [], "need at least 2 PSFs to estimate t")
    ref = psfs[0]
    d0 = ref.distance_mm
    meas = []
    for p in psfs[1:]:
        s, c = relative_scale(ref.image, p.image)
        meas.append((p.distance_mm, s, c))
        log(f"  scale {d0:.1f} -> {p.distance_mm:.1f} mm: s={s:.4f} (corr {c:.2f})")
    d = np.array([m[0] for m in meas])
    s = np.array([m[1] for m in meas])
    w = np.clip(np.array([m[2] for m in meas]), 0.05, None)
    ts = np.linspace(0.01, 0.95 * d0, 6000)
    model = (d[None, :] / (d[None, :] - ts[:, None])) * ((d0 - ts[:, None]) / d0)
    sse = ((model - s[None, :]) ** 2 * w[None, :]).sum(axis=1)
    k = int(np.argmin(sse))
    rms = float(np.sqrt(sse[k] / w.sum()))
    note = "ok"
    if np.median(w) < 0.3:
        note = "low correlation between PSFs - estimate unreliable, set t manually"
    elif k in (0, len(ts) - 1):
        note = "fit hit the search bound - estimate unreliable, set t manually"
    return TEstimate(float(ts[k]), rms, meas, note)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass
class Prediction:
    image: np.ndarray          # float32 counts/us on the sensor
    known: np.ndarray          # bool: False where the pattern came from outside the capture
    info: dict


class PsfModel:
    def __init__(self, run_dir: Path, t_mm: float | None = None,
                 pixel_pitch_um: float = PIXEL_PITCH_UM, log=print):
        self.run_dir = Path(run_dir)
        self.manifest, self.psfs = load_run(self.run_dir)
        self.pitch_mm = pixel_pitch_um / 1000.0
        self.h, self.w = self.psfs[0].image.shape
        if t_mm is None:
            log(f"Estimating diffuser-to-sensor distance t from {len(self.psfs)} PSFs...")
            self.t_estimate = estimate_t(self.psfs, log)
            if self.t_estimate.t_mm is None:
                raise ValueError("Can't estimate t (" + self.t_estimate.note + "); pass t_mm")
            log(f"  t = {self.t_estimate.t_mm:.3f} mm (rms residual {self.t_estimate.rms_residual:.4f}, "
                f"{self.t_estimate.note})")
        else:
            self.t_estimate = TEstimate(None, None, [], "t given explicitly")
        self.t_mm = float(t_mm if t_mm is not None else self.t_estimate.t_mm)

    @property
    def depth_range(self) -> tuple[float, float]:
        return self.psfs[0].distance_mm, self.psfs[-1].distance_mm

    def nearest(self, d_mm: float) -> CapturedPsf:
        return min(self.psfs, key=lambda p: abs(1 / p.distance_mm - 1 / d_mm))

    def _check(self, d_mm: float, t: float):
        if not d_mm > t:
            raise ValueError(f"depth {d_mm} mm must be greater than t = {t:.3f} mm")

    def shift_px(self, x_mm: float, y_mm: float, d_mm: float, t: float) -> tuple[float, float]:
        k = -t / (d_mm - t) / self.pitch_mm
        return k * x_mm, k * y_mm

    def _depth_transform(self, d_mm: float, t: float):
        ref = self.nearest(d_mm)
        s = magnification(d_mm, t) / magnification(ref.distance_mm, t)
        bright = ((ref.distance_mm - t) / (d_mm - t)) ** 2 / s ** 2
        return ref, s, bright

    def predict(self, x_mm: float, y_mm: float, d_mm: float, t_mm: float | None = None) -> Prediction:
        t = self.t_mm if t_mm is None else float(t_mm)
        self._check(d_mm, t)
        ref, s, bright = self._depth_transform(d_mm, t)
        dx, dy = self.shift_px(x_mm, y_mm, d_mm, t)
        cx, cy = ref.centroid
        A = _scale_matrix(s, cx, cy, dx, dy)
        img = cv2.warpAffine(ref.image, A, (self.w, self.h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) * bright
        known = cv2.warpAffine(np.ones_like(ref.image), A, (self.w, self.h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
        full_energy = float(ref.image.sum()) * bright * s * s   # all captured light, scaled
        on_sensor = float(img.sum())
        peak = float(np.percentile(img[known], 99.999)) if known.any() else 0.0
        d_lo, d_hi = self.depth_range
        info = {
            "x_mm": x_mm, "y_mm": y_mm, "d_mm": d_mm, "t_mm": t,
            "z_from_diffuser_mm": d_mm - t,
            "nearest_capture": {"index": ref.index, "distance_mm": ref.distance_mm, "file": ref.file,
                                "dark_subtracted": ref.dark_subtracted, "warnings": ref.warnings},
            "scale_from_capture": s,
            "magnification": magnification(d_mm, t),
            "shift_px": [dx, dy],
            "scale_center_px": [cx, cy],
            "captured_energy_on_sensor": on_sensor / full_energy if full_energy > 0 else 0.0,
            "unknown_area_frac": float(1 - known.mean()),
            "peak_counts_per_us": peak,
            "exposure_for_75pct_us": (0.75 * SENSOR_MAX / peak) if peak > 0 else None,
            "extrapolated_depth": not (d_lo - 1e-6 <= d_mm <= d_hi + 1e-6),
        }
        return Prediction(img.astype(np.float32), known, info)

    def fov_map(self, d_mm: float, t_mm: float | None = None, nx: int = 161, ny: int = 101):
        """Fraction of the captured PSF energy that lands on the sensor, for points
        across the object plane at depth d. Returns (map[ny, nx], x_max_mm, y_max_mm)."""
        t = self.t_mm if t_mm is None else float(t_mm)
        self._check(d_mm, t)
        ref, s, _ = self._depth_transform(d_mm, t)
        cx, cy = ref.centroid
        scaled = cv2.warpAffine(ref.image, _scale_matrix(s, cx, cy), (self.w, self.h),
                                flags=cv2.INTER_LINEAR).astype(np.float64)
        total = scaled.sum()
        if total <= 0:
            return np.zeros((ny, nx)), 1.0, 1.0
        ii = cv2.integral(scaled)       # (h+1, w+1)
        mm_per_px = self.pitch_mm * (d_mm - t) / t     # object mm per pixel of shift
        x_max, y_max = self.w * mm_per_px, self.h * mm_per_px
        xs = np.linspace(-x_max, x_max, nx)
        ys = np.linspace(-y_max, y_max, ny)
        k = -t / (d_mm - t) / self.pitch_mm
        # pattern pixel u lands at u + shift; it's on the sensor if 0 <= u + shift < size
        sx = np.clip(np.round(k * xs).astype(int), -self.w, self.w)
        sy = np.clip(np.round(k * ys).astype(int), -self.h, self.h)
        x0, x1 = np.clip(-sx, 0, self.w), np.clip(self.w - sx, 0, self.w)
        y0, y1 = np.clip(-sy, 0, self.h), np.clip(self.h - sy, 0, self.h)
        Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
        Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
        energy = ii[Y1, X1] - ii[Y0, X1] - ii[Y1, X0] + ii[Y0, X0]
        return energy / total, x_max, y_max


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def render(img: np.ndarray, known: np.ndarray | None = None, gamma: float = 0.5,
           scale: float = 0.6, hi: float | None = None) -> np.ndarray:
    """float image -> 8-bit BGR. Unknown region tinted blue."""
    vals = img[known] if known is not None and known.any() else img
    hi = hi or max(float(np.percentile(vals, 99.9)), 1e-9)
    g = np.clip(img / hi, 0, 1) ** gamma
    bgr = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if known is not None and not known.all():
        bgr[~known] = (bgr[~known] * 0.3 + np.array([90, 40, 20]) * 0.7).astype(np.uint8)
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr
