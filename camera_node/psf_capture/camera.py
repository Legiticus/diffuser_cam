"""Camera access: the real OV9281 through picamera2, or a synthetic fake.

  Frame        - one raw 10-bit frame (stored as uint16) plus what the camera reported
  frame_stats  - histogram / saturation / peak summary used by the Live tab and exposure control
  RealCamera   - picamera2; a background thread keeps the latest raw frame and preview
  FakeWorld    - simulated source position (x, y from the session, z from the fake stage) and light
  FakeCamera   - renders a caustic-like PSF that shifts and scales with the simulated source

Both cameras offer: set_controls(exposure_us, gain), grab(n, after_ns) -> [Frame],
preview_jpeg() -> bytes, latest() -> Frame | None, plus black_level and *_limits.
Auto exposure / gain is never enabled.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import (BIT_DEPTH, DIFFUSER_SENSOR_MM, PIXEL_PITCH_MM, PREVIEW_SIZE, SATURATION,
                     SENSOR_SIZE, SHIFT_SIGN_COL, SHIFT_SIGN_ROW, Z_HOME_MM)

log = logging.getLogger(__name__)

EXPOSURE_LIMITS_US = (9, 10_713_782)    # what the OV9281 driver reports
GAIN_LIMITS = (1.0, 15.9375)
GAIN_STEP = 1 / 16


@dataclass
class Frame:
    raw: np.ndarray            # uint16 (height, width), raw values 0..1023, unchanged
    exposure_us: int           # as reported by the camera for this frame
    gain: float                # as reported by the camera for this frame
    requested_exposure_us: int
    timestamp_ns: int          # CLOCK_MONOTONIC, start of exposure


def robust_peak(raw: np.ndarray, k: int = 10) -> int:
    """k-th brightest pixel: the 'brightest pixel' ignoring a few hot pixels."""
    flat = raw.ravel()
    return int(np.partition(flat, flat.size - k)[flat.size - k])


def frame_stats(raw: np.ndarray, black_level: int, bins: int = 128) -> dict:
    hist = np.bincount((raw.ravel() >> (BIT_DEPTH - 7)).astype(np.intp), minlength=bins)[:bins]
    saturated = int(np.count_nonzero(raw >= SATURATION))
    return {
        "histogram": hist.tolist(),
        "saturated_px": saturated,
        "saturated_pct": 100.0 * saturated / raw.size,
        "peak": robust_peak(raw),
        "max": int(raw.max()),
        "mean": float(raw.mean()),
        "black_level": black_level,
        "saturation": SATURATION,
    }


def clamp_controls(exposure_us: float, gain: float) -> tuple[int, float]:
    exposure = int(min(max(round(exposure_us), EXPOSURE_LIMITS_US[0]), EXPOSURE_LIMITS_US[1]))
    gain = min(max(round(gain / GAIN_STEP) * GAIN_STEP, GAIN_LIMITS[0]), GAIN_LIMITS[1])
    return exposure, gain


def jpeg(gray8: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray8, "L").save(buf, "JPEG", quality=80)
    return buf.getvalue()


class RealCamera:
    exposure_limits = EXPOSURE_LIMITS_US
    gain_limits = GAIN_LIMITS

    def __init__(self, exposure_us: int, gain: float):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        self.exposure_us, self.gain = clamp_controls(exposure_us, gain)
        cfg = self.picam2.create_video_configuration(
            main={"size": PREVIEW_SIZE, "format": "YUV420"},     # preview only
            raw={"format": "R10", "size": SENSOR_SIZE},          # data: unpacked 10-bit in uint16
            buffer_count=4,
            controls={"AeEnable": False, **self._controls()},
        )
        self.picam2.configure(cfg)
        # Only valid after configure() (before, the limit follows the default frame rate).
        self.exposure_limits = tuple(int(v) for v in self.picam2.camera_controls["ExposureTime"][:2])
        self.gain_limits = tuple(float(v) for v in self.picam2.camera_controls["AnalogueGain"][:2])
        self.black_level = 0
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._preview: np.ndarray | None = None
        self._seq = 0
        self._running = True
        self.picam2.start()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        if not self.wait_frame(timeout=5):
            raise RuntimeError("Camera produced no frames")

    def _controls(self) -> dict:
        frame_us = max(33333, self.exposure_us + 1000)
        return {"ExposureTime": self.exposure_us, "AnalogueGain": self.gain,
                "FrameDurationLimits": (frame_us, frame_us)}

    def _loop(self):
        w, h = SENSOR_SIZE
        pw, ph = PREVIEW_SIZE
        while self._running:
            try:
                req = self.picam2.capture_request()
                try:
                    raw = req.make_array("raw").view("<u2")[:h, :w].copy()
                    preview = req.make_array("main")[:ph, :pw].copy()   # Y plane of YUV420
                    md = req.get_metadata()
                finally:
                    req.release()
            except Exception:
                if self._running:
                    log.exception("camera capture failed")
                    time.sleep(0.5)
                continue
            frame = Frame(raw, int(md["ExposureTime"]), float(md["AnalogueGain"]), self.exposure_us,
                          int(md.get("SensorTimestamp", time.monotonic_ns())))
            with self._cond:
                self.black_level = int(md.get("SensorBlackLevels", (0,))[0]) >> (16 - BIT_DEPTH)
                self._latest, self._preview = frame, preview
                self._seq += 1
                self._cond.notify_all()

    def wait_frame(self, timeout: float) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._latest is not None, timeout)

    def set_controls(self, exposure_us: float, gain: float) -> None:
        self.exposure_us, self.gain = clamp_controls(exposure_us, gain)
        self.picam2.set_controls(self._controls())

    def _matches(self, f: Frame) -> bool:
        # The sensor quantises exposure to whole lines (~10 us).
        return (abs(f.exposure_us - self.exposure_us) <= 12 + 0.001 * self.exposure_us
                and abs(f.gain - self.gain) < GAIN_STEP / 2)

    def grab(self, n: int, after_ns: int | None = None) -> list[Frame]:
        """n new frames that started after `after_ns` with the requested controls applied
        (control changes take ~7 frames to reach the sensor)."""
        after_ns = time.monotonic_ns() if after_ns is None else after_ns
        frame_s = max(0.0334, self.exposure_us / 1e6)
        deadline = time.monotonic() + 5 + (n + 12) * frame_s * 1.5
        frames: list[Frame] = []
        with self._cond:
            seq = self._seq
            while len(frames) < n:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"Got {len(frames)}/{n} frames at {self.exposure_us} us")
                if not self._cond.wait_for(lambda: self._seq > seq, timeout=min(left, 3 * frame_s + 1)):
                    continue
                seq = self._seq
                f = self._latest
                if f.timestamp_ns > after_ns and self._matches(f):
                    frames.append(f)
        return frames

    def latest(self) -> Frame | None:
        with self._cond:
            return self._latest

    def preview_jpeg(self) -> bytes | None:
        with self._cond:
            preview = self._preview
        return None if preview is None else jpeg(preview)

    def close(self):
        self._running = False
        self._thread.join(timeout=5)
        self.picam2.stop()
        self.picam2.close()


class FakeWorld:
    """What the fake camera 'sees'. z comes from the fake stage; x, y are set when a
    session starts (in reality the user moves the source by hand)."""

    def __init__(self, stage_fw=None, x_mm: float = 0.0, y_mm: float = 0.0):
        self.stage_fw = stage_fw
        self.x_mm, self.y_mm = x_mm, y_mm
        self.light_on = True
        self.jitter_px = 0.3          # simulated vibration between frames
        self.brightness = 5.0         # peak counts per us at z = 10 mm, gain 1
        self.ambient = 2e-5           # counts per us everywhere (stray light + dark current)

    @property
    def z_mm(self) -> float:
        return self.stage_fw.z if self.stage_fw is not None else 100.0


def caustic_texture(size: int = 1024, seed: int = 1) -> np.ndarray:
    """Network of bright curves (zero crossings of a smooth random field) inside a
    round aperture, i.e. something that looks like a diffuser PSF."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    fr = np.hypot(f[:, None], f[None, :])
    field = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * np.exp(-(fr / 0.012) ** 2)).real
    field /= field.std()
    lines = np.exp(-(field / 0.12) ** 2)
    r = np.hypot(*np.meshgrid(np.arange(size) - size / 2, np.arange(size) - size / 2))
    aperture = np.clip((0.42 * size - r) / 20, 0, 1)
    return (lines * aperture).astype(np.float32)


class FakeCamera:
    exposure_limits = EXPOSURE_LIMITS_US
    gain_limits = GAIN_LIMITS

    def __init__(self, world: FakeWorld, exposure_us: int = 10000, gain: float = 1.0,
                 time_scale: float = 0.0, seed: int = 0):
        self.world = world
        self.exposure_us, self.gain = clamp_controls(exposure_us, gain)
        self.black_level = 64
        self.time_scale = time_scale          # 1.0 = sleep for the real exposure time
        self.texture = caustic_texture()
        self.rng = np.random.default_rng(seed)
        self._latest: Frame | None = None

    def set_controls(self, exposure_us: float, gain: float) -> None:
        self.exposure_us, self.gain = clamp_controls(exposure_us, gain)

    def expected_shift_px(self, x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float]:
        """(row, col) shift of the synthetic PSF on the sensor for a source at x, y, z."""
        k = DIFFUSER_SENSOR_MM / z_mm / PIXEL_PITCH_MM
        return SHIFT_SIGN_ROW * y_mm * k, SHIFT_SIGN_COL * x_mm * k

    def render(self, x_mm: float, y_mm: float, z_mm: float, jitter=(0.0, 0.0)) -> np.ndarray:
        """Noise-free signal in counts per microsecond at gain 1."""
        w, h = SENSOR_SIZE
        size = self.texture.shape[0]
        mag = 0.75 * (1 + DIFFUSER_SENSOR_MM / z_mm)     # PSF scale ~ 1 + d/z
        dr, dc = self.expected_shift_px(x_mm, y_mm, z_mm)
        rows = (np.arange(h) - h / 2 - dr - jitter[0]) / mag + size / 2
        cols = (np.arange(w) - w / 2 - dc - jitter[1]) / mag + size / 2
        img = _bilinear(self.texture, rows, cols)
        return img * (self.world.brightness * (Z_HOME_MM / z_mm) ** 2)

    def _frame(self) -> Frame:
        wd = self.world
        rate = np.full((SENSOR_SIZE[1], SENSOR_SIZE[0]), wd.ambient, np.float32)
        if wd.light_on:
            jitter = self.rng.normal(0, wd.jitter_px, 2) if wd.jitter_px else (0.0, 0.0)
            rate += self.render(wd.x_mm, wd.y_mm, wd.z_mm, jitter)
        signal = rate * self.exposure_us * self.gain
        noise = self.rng.standard_normal(signal.shape, dtype=np.float32) * np.sqrt(signal + 2.0)
        raw = np.clip(np.rint(signal + noise + self.black_level), 0, SATURATION).astype(np.uint16)
        return Frame(raw, self.exposure_us, self.gain, self.exposure_us, time.monotonic_ns())

    def grab(self, n: int, after_ns: int | None = None) -> list[Frame]:
        frames = []
        for _ in range(n):
            if self.time_scale:
                time.sleep(self.time_scale * max(self.exposure_us / 1e6, 1 / 30))
            frames.append(self._frame())
        self._latest = frames[-1]
        return frames

    def latest(self) -> Frame | None:
        """A recent frame for the live view (rendering is slow, so reuse it for 0.2 s)."""
        if self._latest is None or time.monotonic_ns() - self._latest.timestamp_ns > 2e8:
            self.grab(1)
        return self._latest

    def preview_jpeg(self) -> bytes:
        raw = self.latest().raw[::2, ::2]               # 640x400, like the real preview stream
        return jpeg((np.clip(raw.astype(np.int32) - self.black_level, 0, 1023) >> 2).astype(np.uint8))

    def close(self):
        pass


def _bilinear(img: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sample img on the grid rows x cols (fractional, separable); 0 outside."""
    def axis(coord, n):
        i0 = np.floor(coord).astype(np.intp)
        f = (coord - i0).astype(np.float32)
        ok0 = (i0 >= 0) & (i0 < n)
        ok1 = (i0 + 1 >= 0) & (i0 + 1 < n)
        return np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1), (1 - f) * ok0, f * ok1
    r0, r1, wr0, wr1 = axis(rows, img.shape[0])
    c0, c1, wc0, wc1 = axis(cols, img.shape[1])
    top = img[r0][:, c0] * wc0 + img[r0][:, c1] * wc1
    bot = img[r1][:, c0] * wc0 + img[r1][:, c1] * wc1
    return top * wr0[:, None] + bot * wr1[:, None]
