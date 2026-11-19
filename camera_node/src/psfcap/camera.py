"""Picamera2 wrapper: continuous raw 10-bit capture on a background thread."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .constants import GAIN, SENSOR_SIZE

log = logging.getLogger(__name__)


@dataclass
class Frame:
    seq: int
    raw: np.ndarray          # uint16 (h, w), 0..1023
    timestamp_ns: int        # SensorTimestamp, CLOCK_MONOTONIC
    exposure_us: int


class Camera:
    """Owns the Picamera2 instance. A background thread keeps grabbing raw frames;
    the web stream reads the latest one and the sweep waits for qualifying ones."""

    def __init__(self, exposure_us: int):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (640, 400), "format": "YUV420"},   # required, unused
            raw={"format": "R10", "size": SENSOR_SIZE},
            buffer_count=4,
            controls=self._exposure_controls(exposure_us) | {
                "AeEnable": False,
                "AnalogueGain": GAIN,
                "NoiseReductionMode": 0,
            },
        )
        self.picam2.configure(cfg)
        self.raw_cfg = self.picam2.camera_configuration()["raw"]
        self.requested_exposure = exposure_us
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @staticmethod
    def _exposure_controls(exposure_us: int) -> dict:
        frame_us = max(33333, int(exposure_us) + 2000)
        return {"ExposureTime": int(exposure_us), "FrameDurationLimits": (frame_us, frame_us)}

    def start(self):
        self.picam2.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.picam2.stop()
        self.picam2.close()

    def _loop(self):
        w, h = self.raw_cfg["size"]
        while self._running:
            try:
                req = self.picam2.capture_request()
                try:
                    buf = req.make_array("raw")
                    md = req.get_metadata()
                finally:
                    req.release()
            except Exception as e:  # keep the thread alive, surface the error
                self.error = repr(e)
                log.exception("camera capture failed")
                time.sleep(0.5)
                continue
            # R10 is unpacked: little-endian 16-bit words, values 0..1023.
            raw = buf.view("<u2")[:h, :w]
            with self._cond:
                self._seq += 1
                self._latest = Frame(self._seq, raw, int(md.get("SensorTimestamp", time.monotonic_ns())),
                                     int(md.get("ExposureTime", 0)))
                self._cond.notify_all()

    def latest(self) -> Frame | None:
        with self._cond:
            return self._latest

    def wait_newer(self, seq: int, timeout: float = 2.0) -> Frame | None:
        with self._cond:
            self._cond.wait_for(lambda: self._latest is not None and self._latest.seq > seq, timeout)
            return self._latest

    def set_exposure(self, exposure_us: int):
        self.requested_exposure = int(exposure_us)
        self.picam2.set_controls(self._exposure_controls(exposure_us))

    @staticmethod
    def exposure_matches(actual: int, requested: int) -> bool:
        # The sensor quantises exposure to whole line times (~9 us).
        return abs(actual - requested) <= max(0.02 * requested, 20)

    def grab(self, n: int, not_before_ns: int) -> list[Frame]:
        """Return n consecutive new frames that started after `not_before_ns` and
        were taken with the currently requested exposure."""
        req = self.requested_exposure
        frame_s = max(0.0333, req / 1e6)
        deadline = time.monotonic() + 5.0 + (n + 12) * frame_s * 1.5
        out: list[Frame] = []
        seq = self.latest().seq if self.latest() else 0
        while len(out) < n:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Only got {len(out)}/{n} valid frames (exposure {req} us)")
            f = self.wait_newer(seq, timeout=max(1.0, 3 * frame_s))
            if f is None or f.seq == seq:
                continue
            seq = f.seq
            if f.timestamp_ns > not_before_ns and self.exposure_matches(f.exposure_us, req):
                out.append(f)
        return out
