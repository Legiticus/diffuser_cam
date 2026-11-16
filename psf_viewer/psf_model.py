"""
Predicts the sensor PSF for a point anywhere in front of the camera, from a
psf_capture depth sweep (an on-axis point source captured at a series of depths).

Geometry (DiffuserCam, Antipa et al.): the diffuser sits t in front of the sensor;
a point at source-to-sensor distance d is z = d - t from the diffuser.

  * Depth: the caustic pattern is magnified by M(d) = d / (d - t) about a fixed
    point on the sensor, the scale centre (where the source axis meets the sensor).
  * Lateral: moving the point sideways by (x, y) mm translates the pattern on the
    sensor by -(t / z) * (x, y).

Prediction for (x, y, d):
  1. take the two captures that bracket d (in 1/d; the nearest end capture when d is
     outside the captured range),
  2. scale each by M(d) / M(d_i) about the scale centre and blend them, weighted
     linearly in 1/d (the fine speckle changes with depth, not only its scale),
  3. set the total brightness from a smooth fit of captured energy vs depth (real
     runs don't follow an ideal 1/z^2 law), falling back to 1/z^2 outside the range,
  4. translate by -(t / z) * (x, y) / pixel_pitch.

Parts of the prediction that come from outside the captured sensor area are unknown:
they're flagged in a mask, never invented.

t and the scale centre are estimated from the run: a few PSFs are compared with the
nearest one to measure their relative scale and offset (FFT cross-correlation), t is
fitted to the scale-vs-depth model, and the centre to the offsets. If t is known from
the build it can be given instead.

Object coordinates (x, y) are in mm along the sensor's column (x) and row (y)
directions, measured from the axis the light source travelled along.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

PIXEL_PITCH_UM = 3.0       # OV9281
SENSOR_MAX = 1023


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class Capture:
    index: int
    distance_mm: float
    file: str
    dark_file: str | None
    exposure_us: float
    image: np.ndarray          # float32 counts/us, background-subtracted (not clipped at 0:
                               # clipping noise would bias the energy upward)
    dark_subtracted: bool
    warnings: list = field(default_factory=list)

    @property
    def energy(self) -> float:
        return float(self.image.sum(dtype=np.float64))


def read_manifest(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text())


def capture_key(rec: dict) -> tuple:
    """Identifies a manifest record's data; changes when e.g. a dark frame is added."""
    return (rec["file"], rec.get("dark_file"), float(rec["exposure_us"]), float(rec["distance_mm"]))


def load_capture(run_dir: Path, rec: dict) -> Capture:
    run_dir = Path(run_dir)
    raw = np.load(run_dir / rec["file"]).astype(np.float32)
    dark_file = rec.get("dark_file")
    if dark_file and (run_dir / dark_file).exists():
        img = raw - np.load(run_dir / dark_file).astype(np.float32)
        dark = True
    else:   # no dark frame (yet): estimate a flat background
        img = raw - float(np.percentile(raw[::4, ::4], 1))
        dark = False
    img /= float(rec["exposure_us"])
    return Capture(int(rec["index"]), float(rec["distance_mm"]), rec["file"], dark_file,
                   float(rec["exposure_us"]), img, dark, list(rec.get("warnings", [])))


def load_captures(run_dir: Path, manifest: dict, cache: dict | None = None) -> list[Capture]:
    """Load every PSF in the manifest, sorted by distance. `cache` maps capture_key ->
    Capture and is reused/updated, so a growing run only loads its new files."""
    caps = []
    for rec in manifest.get("psfs", []):
        k = capture_key(rec)
        cap = cache.get(k) if cache is not None else None
        if cap is None:
            cap = load_capture(run_dir, rec)
            if cache is not None:
                cache[k] = cap
        caps.append(cap)
    caps.sort(key=lambda c: c.distance_mm)
    return caps


def centroid(img: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(np.clip(img, 0, None).astype(np.float32))
    if m["m00"] <= 0:
        return (img.shape[1] / 2, img.shape[0] / 2)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def magnification(d_mm: float, t_mm: float) -> float:
    return d_mm / (d_mm - t_mm)


def _affine(s: float, cx: float, cy: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Scale by s about (cx, cy), then translate by (dx, dy)."""
    return np.float32([[s, 0, (1 - s) * cx + dx], [0, s, (1 - s) * cy + dy]])


def _warp(img: np.ndarray, A: np.ndarray) -> np.ndarray:
    h, w = img.shape
    return cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# --------------------------------------------------------------------------- #
# Estimating t (diffuser-to-sensor distance) and the scale centre
# --------------------------------------------------------------------------- #
def _normalise(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def _xcorr(target_f: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Normalised circular cross-correlation; peak at tau means target(x) ~ moving(x - tau)."""
    m = np.fft.rfft2(_normalise(moving))
    return np.fft.irfft2(target_f * np.conj(m), s=moving.shape)


def measure_scale(ref: np.ndarray, target: np.ndarray, lo: float = 0.4, hi: float = 1.02
                  ) -> tuple[float, tuple[float, float], float]:
    """Find s such that `ref` scaled by s about the image centre, then shifted, best
    matches `target`. Returns (s, shift_xy_px, correlation peak)."""
    h, w = ref.shape

    def down(img, f):
        return cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)

    def search(r, t, grid):
        rh, rw = r.shape
        t_f = np.fft.rfft2(_normalise(t))
        return np.array([_xcorr(t_f, _warp(r, _affine(s, rw / 2, rh / 2))).max() for s in grid])

    # coarse-to-fine: quarter res over the full range, half res, then full res
    g = np.arange(lo, hi, 0.004)
    s0 = float(g[int(np.argmax(search(down(ref, 4), down(target, 4), g)))])
    g = np.arange(s0 - 0.006, s0 + 0.00601, 0.001)
    s0 = float(g[int(np.argmax(search(down(ref, 2), down(target, 2), g)))])
    step = 0.0005
    g = np.arange(s0 - 3 * step, s0 + 3.01 * step, step)
    sc = search(ref, target, g)
    k = int(np.argmax(sc))
    s = float(g[k])
    if 0 < k < len(g) - 1:          # parabolic refinement
        a, b, c = sc[k - 1], sc[k], sc[k + 1]
        den = a - 2 * b + c
        if den < 0:
            s += 0.5 * (a - c) / den * step
    corr = _xcorr(np.fft.rfft2(_normalise(target)), _warp(ref, _affine(s, w / 2, h / 2)))
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    shift = ((ix + w / 2) % w - w / 2, (iy + h / 2) % h - h / 2)
    return s, (float(shift[0]), float(shift[1])), float(sc[k])


@dataclass
class Geometry:
    t_mm: float | None
    center_px: tuple[float, float] | None     # scale centre on the sensor
    rms_residual: float | None
    note: str                                  # "ok", or why the estimate is unreliable
    source: str                                # where it came from (this run / given / an earlier run)
    samples: list = field(default_factory=list)   # [distance_mm, scale, shift_x, shift_y, corr]

    @property
    def ok(self) -> bool:
        return self.t_mm is not None and self.note == "ok"

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Geometry":
        c = d.get("center_px")
        return cls(d.get("t_mm"), tuple(c) if c else None, d.get("rms_residual"), d.get("note", ""),
                   d.get("source", ""), d.get("samples", []))


def pick_samples(caps: list[Capture], n_max: int) -> list[Capture]:
    """Captures (other than the nearest) to measure scale against: spread evenly in
    1/d, always including the farthest (it carries the most information about t)."""
    rest = caps[1:]
    if len(rest) <= n_max:
        return rest
    inv = np.array([1 / c.distance_mm for c in rest])
    targets = np.linspace(inv[0], inv[-1], n_max)
    idx = sorted({int(np.argmin(np.abs(inv - v))) for v in targets} | {len(rest) - 1})
    return [rest[i] for i in idx]


def sample_key(caps: list[Capture], n_max: int) -> list:
    """What an estimate from these captures depends on (for caching)."""
    if len(caps) < 2:
        return []
    return [[c.file, c.distance_mm, c.dark_file] for c in [caps[0]] + pick_samples(caps, n_max)]


def estimate_geometry(caps: list[Capture], n_max: int = 6, log=print) -> Geometry:
    if len(caps) < 2:
        return Geometry(None, None, None, "need at least 2 PSFs to estimate t", "this run")
    ref = caps[0]
    d0 = ref.distance_mm
    h, w = ref.image.shape
    samples = []
    for c in pick_samples(caps, n_max):
        s, (sx, sy), corr = measure_scale(ref.image, c.image)
        samples.append([c.distance_mm, s, sx, sy, corr])
        log(f"  {d0:.1f} -> {c.distance_mm:.1f} mm: scale {s:.4f}, offset ({sx:.0f}, {sy:.0f}) px, corr {corr:.2f}")
    S = np.array(samples)
    d, s, wgt = S[:, 0], S[:, 1], np.clip(S[:, 4], 0.05, None)

    # t: weighted least squares of s = M(d)/M(d0) over a grid
    ts = np.linspace(0.01, 0.95 * d0, 6000)
    model = (d[None, :] / (d[None, :] - ts[:, None])) * ((d0 - ts[:, None]) / d0)
    sse = ((model - s[None, :]) ** 2 * wgt[None, :]).sum(axis=1)
    k = int(np.argmin(sse))
    rms = float(np.sqrt(sse[k] / wgt.sum()))
    note = "ok"
    if np.median(wgt) < 0.3:
        note = "low correlation between PSFs - estimate unreliable, set t manually"
    elif k in (0, len(ts) - 1):
        note = "fit hit the search bound - estimate unreliable, set t manually"
    elif 1 - s.min() < 0.01:
        note = "captured depth range too small to measure t reliably"

    # scale centre: scaling about c instead of the image centre o adds an offset (1-s)(c-o)
    a = 1 - s
    if (a * a).sum() > 1e-5:
        off = (a[:, None] * S[:, 2:4]).sum(0) / (a * a).sum()
        center = (w / 2 + float(off[0]), h / 2 + float(off[1]))
    else:
        center = centroid(ref.image)
    return Geometry(float(ts[k]), center, rms, note, "this run", samples)


# --------------------------------------------------------------------------- #
# Brightness vs depth
# --------------------------------------------------------------------------- #
class EnergyCurve:
    """Total captured energy (counts/us) vs depth: a smooth fit of log E against log z
    over the captures (quadratic with >= 4 captures, else linear), so exposure jitter
    on single captures doesn't make the brightness jump. Beyond the captured range it
    continues as 1/z^2 from the end of the fit."""

    def __init__(self, caps: list[Capture], t_mm: float):
        self.t = t_mm
        self.d = np.array([c.distance_mm for c in caps])
        e = np.array([max(c.energy, 1e-12) for c in caps])
        lz = np.log(self.d - t_mm)
        deg = 2 if len(caps) >= 4 else (1 if len(caps) >= 2 else 0)
        self.coef = np.polyfit(lz, np.log(e), deg) if deg else np.array([np.log(e[0]) + 2 * lz[0]])
        self.flat_z2 = deg == 0          # single capture: pure 1/z^2 through it
        self.residual = (np.exp(np.log(e) - self._fit(lz)) - 1) if deg else np.zeros(1)

    def _fit(self, lz):
        if self.flat_z2:
            return self.coef[0] - 2 * lz
        return np.polyval(self.coef, lz)

    def __call__(self, d_mm: float) -> float:
        d_end = float(np.clip(d_mm, self.d.min(), self.d.max()))
        e_end = float(np.exp(self._fit(np.log(d_end - self.t))))
        return e_end * ((d_end - self.t) / (d_mm - self.t)) ** 2


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass
class Prediction:
    image: np.ndarray          # float32 counts/us on the sensor
    known: np.ndarray          # bool: False where the pattern came from outside the capture
    info: dict


class PsfModel:
    def __init__(self, run_dir: Path, manifest: dict, caps: list[Capture], geometry: Geometry,
                 pixel_pitch_um: float = PIXEL_PITCH_UM):
        if not caps:
            raise ValueError("no PSFs captured yet")
        if geometry.t_mm is None:
            raise ValueError(f"t unknown ({geometry.note}); pass --t-mm")
        self.run_dir = Path(run_dir)
        self.manifest = manifest
        self.caps = caps
        self.geometry = geometry
        self.t_mm = float(geometry.t_mm)
        self.center = geometry.center_px or centroid(caps[0].image)
        self.pitch_mm = pixel_pitch_um / 1000.0
        self.h, self.w = caps[0].image.shape
        self._curves: dict[float, EnergyCurve] = {}

    @classmethod
    def from_run(cls, run_dir: Path, t_mm: float | None = None,
                 pixel_pitch_um: float = PIXEL_PITCH_UM, log=print) -> "PsfModel":
        manifest = read_manifest(run_dir)
        caps = load_captures(run_dir, manifest)
        if t_mm is None:
            geo = estimate_geometry(caps, log=log)
        else:
            g = estimate_geometry(caps, log=log) if len(caps) >= 2 else None
            geo = Geometry(t_mm, g.center_px if g else None, None, "ok", "given")
        return cls(run_dir, manifest, caps, geo, pixel_pitch_um)

    @property
    def depth_range(self) -> tuple[float, float]:
        return self.caps[0].distance_mm, self.caps[-1].distance_mm

    def curve(self, t: float) -> EnergyCurve:
        if t not in self._curves:
            if len(self._curves) > 8:
                self._curves.clear()
            self._curves[t] = EnergyCurve(self.caps, t)
        return self._curves[t]

    def _t(self, t_mm):
        t = self.t_mm if t_mm is None else float(t_mm)
        if not (np.isfinite(t) and 0 < t < self.depth_range[0]):
            raise ValueError(f"t must be between 0 and the nearest capture ({self.depth_range[0]:.1f} mm)")
        return t

    def bracket(self, d_mm: float) -> list[tuple[Capture, float]]:
        """Captures used for depth d, with blend weights (linear in 1/d)."""
        inv = 1 / d_mm
        lo, hi = self.depth_range
        if d_mm <= lo:
            return [(self.caps[0], 1.0)]
        if d_mm >= hi:
            return [(self.caps[-1], 1.0)]
        for a, b in zip(self.caps, self.caps[1:]):
            if a.distance_mm <= d_mm <= b.distance_mm:
                ia, ib = 1 / a.distance_mm, 1 / b.distance_mm
                wa = (inv - ib) / (ia - ib) if ia != ib else 1.0
                out = [(a, wa), (b, 1 - wa)]
                return [(c, w) for c, w in out if w > 1e-6]
        raise AssertionError("unreachable")

    def shift_px(self, x_mm: float, y_mm: float, d_mm: float, t: float) -> tuple[float, float]:
        k = -t / (d_mm - t) / self.pitch_mm
        return k * x_mm, k * y_mm

    def predict(self, x_mm: float, y_mm: float, d_mm: float, t_mm: float | None = None) -> Prediction:
        t = self._t(t_mm)
        if not d_mm > t:
            raise ValueError(f"depth {d_mm} mm must be greater than t = {t:.3f} mm")
        dx, dy = self.shift_px(x_mm, y_mm, d_mm, t)
        cx, cy = self.center
        e_target = self.curve(t)(d_mm)
        img = np.zeros((self.h, self.w), np.float32)
        known = np.ones((self.h, self.w), bool)
        used = []
        for cap, wgt in self.bracket(d_mm):
            s = magnification(d_mm, t) / magnification(cap.distance_mm, t)
            A = _affine(s, cx, cy, dx, dy)
            # warping by s multiplies the sum by s^2; normalise to the target energy
            gain = wgt * e_target / (s * s * cap.energy) if cap.energy > 0 else 0.0
            img += _warp(cap.image, A) * np.float32(gain)
            known &= _warp(np.ones_like(cap.image), A) > 0.999
            used.append({"index": cap.index, "distance_mm": cap.distance_mm, "file": cap.file,
                         "weight": wgt, "scale": s, "dark_subtracted": cap.dark_subtracted,
                         "warnings": cap.warnings})
        on_sensor = float(img.sum(dtype=np.float64))
        peak = float(np.percentile(img[known], 99.999)) if known.any() else 0.0
        d_lo, d_hi = self.depth_range
        info = {
            "x_mm": x_mm, "y_mm": y_mm, "d_mm": d_mm, "t_mm": t,
            "z_from_diffuser_mm": d_mm - t,
            "based_on": used,
            "nearest_capture_index": max(used, key=lambda u: u["weight"])["index"],
            "magnification": magnification(d_mm, t),
            "shift_px": [dx, dy],
            "scale_center_px": [cx, cy],
            "total_energy_counts_per_us": e_target,
            "energy_on_sensor_frac": on_sensor / e_target if e_target > 0 else 0.0,
            "unknown_area_frac": float(1 - known.mean()),
            "peak_counts_per_us": peak,
            "exposure_for_75pct_us": (0.75 * SENSOR_MAX / peak) if peak > 0 else None,
            "extrapolated_depth": not (d_lo - 1e-6 <= d_mm <= d_hi + 1e-6),
        }
        return Prediction(img, known, info)

    def fov_map(self, d_mm: float, t_mm: float | None = None, nx: int = 161, ny: int = 101):
        """Fraction of the PSF's light that lands on the sensor, for points across the
        object plane at depth d. Returns (map[ny, nx], x_max_mm, y_max_mm)."""
        t = self._t(t_mm)
        if not d_mm > t:
            raise ValueError(f"depth {d_mm} mm must be greater than t = {t:.3f} mm")
        cap = max(self.bracket(d_mm), key=lambda cw: cw[1])[0]
        s = magnification(d_mm, t) / magnification(cap.distance_mm, t)
        scaled = _warp(cap.image, _affine(s, *self.center)).astype(np.float64)
        total = cap.energy * s * s
        mm_per_px = self.pitch_mm * (d_mm - t) / t     # object mm per pixel of shift
        x_max, y_max = self.w * mm_per_px, self.h * mm_per_px
        if total <= 0:
            return np.zeros((ny, nx)), x_max, y_max
        ii = cv2.integral(scaled)       # (h+1, w+1)
        xs = np.linspace(-x_max, x_max, nx)
        ys = np.linspace(-y_max, y_max, ny)
        k = -1 / mm_per_px
        # pattern pixel u lands at u + shift; it's on the sensor if 0 <= u + shift < size
        sx = np.clip(np.round(k * xs).astype(int), -self.w, self.w)
        sy = np.clip(np.round(k * ys).astype(int), -self.h, self.h)
        x0, x1 = np.clip(-sx, 0, self.w), np.clip(self.w - sx, 0, self.w)
        y0, y1 = np.clip(-sy, 0, self.h), np.clip(self.h - sy, 0, self.h)
        Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
        Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
        energy = ii[Y1, X1] - ii[Y0, X1] - ii[Y1, X0] + ii[Y0, X0]
        return np.clip(energy / total, 0, 1), x_max, y_max


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
def render(img: np.ndarray, known: np.ndarray | None = None, gamma: float = 0.5,
           scale: float = 0.6) -> np.ndarray:
    """float image -> 8-bit BGR. Unknown region tinted blue."""
    vals = img[known] if known is not None and known.any() else img
    hi = max(float(np.percentile(vals, 99.9)), 1e-9)
    g = np.clip(img / hi, 0, 1) ** gamma
    bgr = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if known is not None and not known.all():
        bgr[~known] = (bgr[~known] * 0.3 + np.array([90, 40, 20]) * 0.7).astype(np.uint8)
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bgr
