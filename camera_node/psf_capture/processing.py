"""Turn a session's raw frames into one PSF per z plane. Never modifies raw files.

Steps (each optional, in order):
  1. background  "frames": subtract the mean background frame at the same exposure
                 "black_level": subtract only the sensor black level; "none"
  2. average_frames: mean of the frames at each position within a pass (off: first frame)
  3. passes: combine passes (mean or median), after subpixel phase-correlation
     registration if `register` is on. Passes with low sharpness or correlation can be
     excluded. Pixels with no data after shifting are NaN (the invalid mask) and are
     left out of the combination.
  4. normalize   "exposure": counts per microsecond per unit gain
                 "sum": each plane sums to 1;  "none": counts at the first pass's exposure

Results are cached in <session>/processed/<key>/ (psf.npy: float32 [n_z, h, w], NaN =
invalid; info.json: steps used, measured pass shifts, drift with z and the source
offset it implies). Passes are
always brought to a common exposure before combining because per-position exposure
may differ between passes.

Shared helpers: phase_correlate (subpixel shift between images), shift_image.
"""

from __future__ import annotations

import hashlib
import json
import warnings as _warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from . import storage
from .config import DIFFUSER_SENSOR_MM, PIXEL_PITCH_MM, SHIFT_SIGN_COL, SHIFT_SIGN_ROW

VERSION = 3      # bump when the algorithm changes so old caches are not reused

DEFAULT_STEPS = {
    "background": "frames",
    "average_frames": True,
    "register": True,
    "combine": "mean",
    "exclude": True,
    "min_quality": 0.7,           # exclude a pass below this fraction of the median quality
    "normalize": "exposure",
}


# --- image helpers -------------------------------------------------------------------
def phase_correlate(ref: np.ndarray, img: np.ndarray) -> tuple[float, float, float]:
    """Subpixel shift (dy, dx) such that img(y, x) ~ ref(y - dy, x - dx), plus the
    correlation peak height (0..1, how alike the two are).

    The normalised cross-power spectrum is multiplied by a Gaussian so the peak is a
    smooth Gaussian; a 3-point fit of its logarithm then gives the subpixel position."""
    h, w = ref.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    a = np.nan_to_num(ref - np.nanmean(ref)) * win
    b = np.nan_to_num(img - np.nanmean(img)) * win
    cross = np.fft.rfft2(b) * np.conj(np.fft.rfft2(a))
    cross /= np.abs(cross) + 1e-12
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    sigma = 1.5                                         # px, width of the smoothed peak
    corr = np.fft.irfft2(cross * np.exp(-2 * (np.pi * sigma) ** 2 * (fy ** 2 + fx ** 2)), s=(h, w))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    peak = corr[py, px]

    def refine(c_minus, c0, c_plus):
        lm, l0, lp = (np.log(max(v, 1e-12)) for v in (c_minus, c0, c_plus))
        denom = lm - 2 * l0 + lp
        return 0.0 if denom >= 0 else 0.5 * (lm - lp) / denom

    dy = py + refine(corr[py - 1, px], peak, corr[(py + 1) % h, px])
    dx = px + refine(corr[py, px - 1], peak, corr[py, (px + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    perfect = 1 / (2 * np.pi * sigma ** 2)               # peak height for identical images
    return float(dy), float(dx), float(min(1.0, peak / perfect))


def sample(img: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """img evaluated at fractional source coordinates rows x cols (separable grid),
    bilinear; pixels whose source lies outside the image are NaN."""
    def axis(src, n):
        i0 = np.floor(src).astype(np.intp)
        f = (src - i0).astype(np.float32)
        ok = (src >= 0) & (src <= n - 1)
        return np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1), f, ok

    r0, r1, fr, okr = axis(rows, img.shape[0])
    c0, c1, fc, okc = axis(cols, img.shape[1])
    top = img[r0][:, c0] * (1 - fc) + img[r0][:, c1] * fc
    bot = img[r1][:, c0] * (1 - fc) + img[r1][:, c1] * fc
    out = top * (1 - fr)[:, None] + bot * fr[:, None]
    out[~(okr[:, None] & okc[None, :])] = np.nan
    return out.astype(np.float32)


def shift_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate by (dy, dx) px with bilinear interpolation; uncovered pixels are NaN."""
    h, w = img.shape
    return sample(img, np.arange(h) - dy, np.arange(w) - dx)


def zoom_image(img: np.ndarray, factor: float, center=None) -> np.ndarray:
    """Scale about `center` (row, col; default the image centre); factor > 1 enlarges."""
    h, w = img.shape
    ch, cw = center if center is not None else ((h - 1) / 2, (w - 1) / 2)
    return sample(img, (np.arange(h) - ch) / factor + ch, (np.arange(w) - cw) / factor + cw)


def magnification(z_mm: float, d_mm: float = DIFFUSER_SENSOR_MM) -> float:
    """Approximate PSF scale relative to a source at infinity."""
    return 1 + d_mm / z_mm


def sharpness(img: np.ndarray) -> float:
    """Brightness-independent edge energy; motion blur lowers it."""
    gy, gx = np.gradient(np.nan_to_num(img))
    return float(np.sum(gx ** 2 + gy ** 2) / max(np.sum(np.nan_to_num(img) ** 2), 1e-12))


def shift_per_mm(z_mm: float, d_mm: float = DIFFUSER_SENSOR_MM) -> float:
    """Predicted lateral PSF shift on the sensor per mm of source offset, px/mm."""
    return d_mm / z_mm / PIXEL_PITCH_MM


# --- session processing ------------------------------------------------------------------
def cache_key(meta: dict, steps: dict) -> str:
    blob = json.dumps({"v": VERSION, "steps": steps, "ended": meta.get("ended"),
                       "n": len(meta["frames"])}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def normalise_steps(steps: dict | None) -> dict:
    s = {**DEFAULT_STEPS, **(steps or {})}
    if s["background"] not in ("frames", "black_level", "none"):
        raise ValueError("background must be frames, black_level or none")
    if s["combine"] not in ("mean", "median"):
        raise ValueError("combine must be mean or median")
    if s["normalize"] not in ("exposure", "sum", "none"):
        raise ValueError("normalize must be exposure, sum or none")
    s["min_quality"] = float(s["min_quality"])
    for k in ("average_frames", "register", "exclude"):
        s[k] = bool(s[k])
    return s


def process_session(session_dir: Path, steps: dict | None = None, progress=None) -> Path:
    """Process (or reuse the cached result) and return the cache folder."""
    session_dir = Path(session_dir)
    meta = storage.load_metadata(session_dir)
    if meta["status"] == "running":
        raise ValueError(f"{session_dir.name} is still being captured")
    steps = normalise_steps(steps)
    out = session_dir / "processed" / cache_key(meta, steps)
    if (out / "info.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    z_list = meta["plan"]["actual_z"]
    h, w = meta["camera"]["height"], meta["camera"]["width"]
    psf = np.lib.format.open_memmap(out / "psf.npy", "w+", np.float32, (len(z_list), h, w))
    backgrounds = _BackgroundCache(session_dir, meta, steps["background"], warnings)
    by_position: dict[tuple[int, int], list[dict]] = {}
    for f in meta["frames"]:
        by_position.setdefault((f["z_index"], f["pass"]), []).append(f)
    ref_scale = None
    shifts, no_psf = [], []

    for i, z in enumerate(z_list):
        if progress:
            progress(i / len(z_list), f"{session_dir.name}: z {i + 1}/{len(z_list)}")
        passes = []                                       # (pass no, image per (us * gain))
        for (zi, p), frames in sorted(by_position.items()):
            if zi != i:
                continue
            frames = sorted(frames, key=lambda f: f["frame"])
            if not steps["average_frames"]:
                frames = frames[:1]
            imgs = [storage.read_frame(session_dir / f["file"]).astype(np.float32)
                    - backgrounds.get(f["requested_exposure_us"]) for f in frames]
            scale = float(np.mean([f["exposure_us"] * f["gain"] for f in frames]))
            ref_scale = ref_scale or scale
            passes.append((p, np.mean(imgs, axis=0) / scale))
        if not passes:
            psf[i] = np.nan
            no_psf.append(True)
            continue
        plane, info = _combine_passes(passes, steps)
        if info:
            shifts.append({"z_index": i, "z_mm": z, **info})
        no_psf.append(False)
        psf[i] = plane

    if ref_scale is None:
        raise ValueError(f"{session_dir.name} has no PSF frames")
    for i in range(len(z_list)):                          # normalisation (step 4)
        if no_psf[i]:
            continue
        if steps["normalize"] == "sum":
            psf[i] /= np.nansum(psf[i])
        elif steps["normalize"] == "none":
            psf[i] *= ref_scale
    psf.flush()

    if progress:
        progress(1.0, f"{session_dir.name}: drift")
    drift, drift_fit = _drift_with_z(psf, z_list, no_psf, meta["x_mm"], meta["y_mm"])
    info = {
        "session": session_dir.name, "steps": steps, "processing_version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "x_mm": meta["x_mm"], "y_mm": meta["y_mm"], "plan": {k: v for k, v in meta["plan"].items()},
        "z_mm": z_list, "no_psf": no_psf, "shape": [len(z_list), h, w],
        "units": {"exposure": "counts / (us * gain)", "sum": "fraction of total",
                  "none": f"counts at {ref_scale:.0f} us*gain"}[steps["normalize"]],
        "invalid": "NaN pixels in psf.npy have no data (outside the sensor after registration)",
        "pass_shifts": shifts, "drift": drift, "drift_fit": drift_fit, "warnings": warnings,
    }
    storage.write_json(out / "info.json", info)
    return out


class _BackgroundCache:
    """Mean background frame per requested exposure (or the black level)."""

    def __init__(self, session_dir, meta, mode, warnings):
        self.dir, self.meta, self.mode, self.warnings = session_dir, meta, mode, warnings
        self.cache: dict[int, np.ndarray | float] = {}

    def get(self, exposure_us: int):
        black = float(self.meta["camera"]["black_level"])
        if self.mode == "none":
            return 0.0
        if self.mode == "black_level":
            return black
        if exposure_us not in self.cache:
            files = [b["file"] for b in self.meta["backgrounds"] if b["requested_exposure_us"] == exposure_us]
            if files:
                self.cache[exposure_us] = np.mean(
                    [storage.read_frame(self.dir / f).astype(np.float32) for f in files], axis=0)
            else:
                self.warnings.append(f"No background frames at {exposure_us} us; "
                                     "subtracted the black level instead")
                self.cache[exposure_us] = black
        return self.cache[exposure_us]


def _combine_passes(passes: list[tuple[int, np.ndarray]], steps: dict):
    if len(passes) == 1:
        return passes[0][1], None
    imgs = [img for _, img in passes]
    sharp = [sharpness(img) for img in imgs]
    info = {"passes": []}
    if steps["register"]:
        ref = int(np.argmax(sharp))
        measured = [phase_correlate(imgs[ref], img) for img in imgs]
        corr = [m[2] for m in measured]
        corr[ref] = float(np.median([c for k, c in enumerate(corr) if k != ref]))
        keep = [True] * len(imgs)
        if steps["exclude"]:
            ms, mc = np.median(sharp), np.median(corr)
            keep = [s >= steps["min_quality"] * ms and c >= steps["min_quality"] * mc
                    for s, c in zip(sharp, corr)]
            keep[ref] = True
        kept = [m for m, k in zip(measured, keep) if k]
        mean_dy = float(np.mean([m[0] for m in kept]))
        mean_dx = float(np.mean([m[1] for m in kept]))
        # Shift every pass onto the mean position: random jitter averages out while a
        # shift common to all passes (a real source offset) is kept.
        aligned = []
        for (p, img), (dy, dx, c), k, s in zip(passes, measured, keep, sharp):
            info["passes"].append({"pass": p, "dy": dy - mean_dy, "dx": dx - mean_dx,
                                   "correlation": c, "sharpness": s, "excluded": not k})
            if k:
                aligned.append(shift_image(img, -(dy - mean_dy), -(dx - mean_dx)))
        jitter = [np.hypot(q["dy"], q["dx"]) for q in info["passes"] if not q["excluded"]]
        info["jitter_rms_px"] = float(np.sqrt(np.mean(np.square(jitter))))
    else:
        aligned = imgs
        info["passes"] = [{"pass": p, "dy": 0.0, "dx": 0.0, "sharpness": s, "excluded": False}
                          for (p, _), s in zip(passes, sharp)]
    stack = np.stack(aligned)
    with _warnings.catch_warnings():                       # all-NaN pixels stay NaN
        _warnings.simplefilter("ignore", RuntimeWarning)
        plane = np.nanmedian(stack, axis=0) if steps["combine"] == "median" else np.nanmean(stack, axis=0)
    return plane.astype(np.float32), info


def _drift_with_z(psf, z_list, no_psf, x_mm, y_mm):
    """How the PSF moves with z, and what that says about the source position.

    A source at lateral offset x shifts the PSF by SIGN * x * d / (z * pitch), and the
    pixel of the optical axis is unknown, so the PSF position follows c + B / z.
    B gives the apparent source offset (a constant misplacement and a rail tilt look
    the same here, so only this effective offset is measurable). Per z, the source
    position implied by the measured PSF position minus the entered x, y is the
    correction for that z; on a straight rail it is the same at every z."""
    valid = [i for i in range(len(z_list)) if not no_psf[i]]
    if len(valid) < 2:
        return [], None
    pos = {valid[0]: (0.0, 0.0)}                    # PSF position relative to the first plane
    for a, b in zip(valid, valid[1:]):
        # Undo the predicted change of scale (~1 + d/z) first, otherwise neighbouring
        # caustics decorrelate. Zoom about the centre of the PSF's footprint (the
        # scaling centre) so the zoom itself does not move the PSF. A PSF clipped by
        # the sensor edge biases that centre and with it the apparent offset.
        r = magnification(z_list[a]) / magnification(z_list[b])
        plane = np.asarray(psf[b])
        dy, dx, _ = phase_correlate(_bin2(psf[a]), _bin2(zoom_image(plane, r, _support_center(plane))))
        pos[b] = (pos[a][0] + 2 * dy, pos[a][1] + 2 * dx)

    zs = np.array([z_list[i] for i in valid])
    rows = np.array([pos[i][0] for i in valid])
    cols = np.array([pos[i][1] for i in valid])
    design = np.stack([np.ones_like(zs), 1 / zs], axis=1)
    (c_row, b_row), *_ = np.linalg.lstsq(design, rows, rcond=None)
    (c_col, b_col), *_ = np.linalg.lstsq(design, cols, rcond=None)
    k = shift_per_mm(1.0)                           # px per mm at z = 1 mm (k(z) = k / z)
    apparent = {"x_mm": float(b_col / (SHIFT_SIGN_COL * k)), "y_mm": float(b_row / (SHIFT_SIGN_ROW * k))}
    drift = []
    for i, z, r_px, c_px in zip(valid, zs, rows, cols):
        drift.append({"z_index": i, "z_mm": float(z), "dy_px": float(r_px), "dx_px": float(c_px),
                      "source_dx_mm": float((c_px - c_col) / (SHIFT_SIGN_COL * k / z) - x_mm),
                      "source_dy_mm": float((r_px - c_row) / (SHIFT_SIGN_ROW * k / z) - y_mm)})
    fit_rms = float(np.sqrt(np.mean((rows - design @ [c_row, b_row]) ** 2
                                    + (cols - design @ [c_col, b_col]) ** 2)))
    summary = {"apparent_offset_mm": apparent,
               "offset_error_mm": {"x_mm": apparent["x_mm"] - x_mm, "y_mm": apparent["y_mm"] - y_mm},
               "drift_px": [float(rows[-1] - rows[0]), float(cols[-1] - cols[0])],
               "fit_rms_px": fit_rms}
    return drift, summary


def _support_center(img: np.ndarray, block: int = 16) -> tuple[float, float]:
    """Centre (row, col) of the area the PSF covers: the projected diffuser aperture,
    which scales about the source's projection. Uses a block-averaged image thresholded
    at 30 % of its maximum, so the caustic detail inside does not bias it."""
    h, w = img.shape
    hb, wb = h // block, w // block
    small = np.nan_to_num(img[: hb * block, : wb * block]).reshape(hb, block, wb, block).mean(axis=(1, 3))
    mask = small > 0.3 * small.max() if small.max() > 0 else np.ones_like(small, bool)
    rr, cc = np.nonzero(mask)
    return (rr.mean() + 0.5) * block - 0.5, (cc.mean() + 0.5) * block - 0.5


def _bin2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(img[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2), axis=(1, 3))


def load(cache_dir: Path) -> tuple[np.ndarray, dict]:
    """Memory-mapped PSF stack and its info."""
    return (np.load(Path(cache_dir) / "psf.npy", mmap_mode="r"),
            json.loads((Path(cache_dir) / "info.json").read_text()))
