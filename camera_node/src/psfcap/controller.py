"""Arduino stepper stage over USB serial (ZERO / MOVE protocol), plus a simulator."""

from __future__ import annotations

import glob
import logging
import math
import threading
import time

import numpy as np

from .constants import MAX_TRAVEL_MM

log = logging.getLogger(__name__)


class ControllerError(RuntimeError):
    pass


class LimitHit(ControllerError):
    def __init__(self, which: str, msg: str):
        super().__init__(msg)
        self.which = which  # "front" | "back"


class StageController:
    """Tracks the light-source position in whole motor steps from zero."""

    def __init__(self, port: str, baud: int, steps_per_mm: float, away_sign: int,
                 zero_timeout: float, move_speed_mm_s: float, max_travel_mm: float):
        self.port = port
        self.baud = baud
        self.steps_per_mm = steps_per_mm
        self.away_sign = away_sign
        self.zero_timeout = zero_timeout
        self.move_speed = move_speed_mm_s
        # Software travel limit. This is the protection against running off the
        # back of the rail; the back limit switch may not exist, so it's never relied on.
        self.max_steps = int(math.floor(max_travel_mm * steps_per_mm + 1e-6))
        self.position_steps: int | None = None   # None = unknown, needs ZERO
        self.lock = threading.Lock()
        self._ser = None

    # -- connection --
    def open(self):
        import serial  # pyserial
        log.info("Opening controller on %s @ %d", self.port, self.baud)
        self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        # Opening the port toggles DTR, which resets the Nano. Wait for the
        # bootloader to finish, then drop anything it printed.
        time.sleep(2.5)
        self._ser.reset_input_buffer()

    def close(self):
        if self._ser:
            self._ser.close()
            self._ser = None

    # -- low-level --
    def _command(self, cmd: str, ok: str, timeout: float) -> str:
        assert self._ser is not None
        self._ser.reset_input_buffer()
        log.debug("-> %s", cmd)
        self._ser.write((cmd + "\n").encode("ascii"))
        self._ser.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self._ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            log.debug("<- %s", line)
            if line.startswith(ok):
                return line
            if "FAIL" in line or line.startswith("ERROR"):
                if "Front Limit" in line:
                    raise LimitHit("front", line)
                if "Back Limit" in line:   # only if a back switch is ever fitted
                    raise LimitHit("back", line)
                raise ControllerError(line)
            log.info("controller: %s", line)
        raise ControllerError(f"Timeout after {timeout:.0f}s waiting for '{ok}' to '{cmd}'")

    # -- high-level --
    def zero(self):
        with self.lock:
            self.position_steps = None
            self._command("ZERO", "ZERO_SUCCESS", self.zero_timeout)
            self.position_steps = 0

    def mm_to_steps(self, travel_mm: float) -> int:
        return int(round(travel_mm * self.steps_per_mm))

    def move_to_steps(self, target_steps: int):
        """Absolute move (in steps away from zero) with exact step bookkeeping."""
        with self.lock:
            if self.position_steps is None:
                raise ControllerError("Position unknown: run ZERO first")
            if not 0 <= target_steps <= self.max_steps:
                raise ControllerError(
                    f"Target {target_steps / self.steps_per_mm:.3f} mm travel is outside the "
                    f"allowed range 0..{self.max_steps / self.steps_per_mm:.3f} mm; move refused")
            delta = target_steps - self.position_steps
            if delta == 0:
                return
            fw_steps = self.away_sign * delta          # steps in firmware's frame
            # Firmware does long(800.0f * atof(arg)) which truncates toward zero.
            # Add half a step so float rounding still lands on the exact count.
            arg = (fw_steps + math.copysign(0.5, fw_steps)) / self.steps_per_mm
            dist_mm = abs(delta) / self.steps_per_mm
            timeout = 15.0 + dist_mm / self.move_speed
            try:
                self._command(f"MOVE {arg:.6f}", "MOVE_SUCCESS", timeout)
            except ControllerError:
                self.position_steps = None   # stage stopped somewhere unknown
                raise
            self.position_steps = target_steps

    @property
    def travel_mm(self) -> float | None:
        if self.position_steps is None:
            return None
        return self.position_steps / self.steps_per_mm


class SimulatedSerial:
    """Stands in for the Arduino so the pipeline can be tested without hardware."""

    def __init__(self, steps_per_mm: float, travel_mm: float = MAX_TRAVEL_MM,
                 back_limit_switch: bool = False):
        self.spm = steps_per_mm
        self.pos = None
        self.travel = int(travel_mm * steps_per_mm)
        self.back_switch = back_limit_switch
        self._out: list[bytes] = []

    def reset_input_buffer(self):
        self._out.clear()

    def flush(self):
        pass

    def close(self):
        pass

    def write(self, data: bytes):
        parts = data.decode().strip().split()
        cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
        if cmd == "ZERO":
            time.sleep(1.0)
            self.pos = 0
            self._out.append(b"ZERO_SUCCESS\r\n")
        elif cmd == "MOVE" and arg is not None:
            steps = int(np.float32(self.spm) * np.float32(float(arg)))  # emulate AVR
            time.sleep(min(abs(steps) / self.spm / 5.0, 5.0))
            new = self.pos - steps  # firmware FORWARD = toward front limit
            if new < 0:
                self.pos = 0
                self._out.append(b"MOVE_FAIL - Front Limit Triggered\r\n")
            elif new > self.travel:
                self.pos = self.travel
                if self.back_switch:
                    self._out.append(b"MOVE_FAIL - Back Limit Triggered\r\n")
                else:   # no switch: carriage stalls at the hard stop, firmware still says OK
                    log.error("SIM: carriage driven into the back hard stop!")
                    self._out.append(b"MOVE_SUCCESS\r\n")
            else:
                self.pos = new
                self._out.append(b"MOVE_SUCCESS\r\n")
        else:
            self._out.append(b"ERROR - Not a valid command\r\n")

    def readline(self) -> bytes:
        if self._out:
            return self._out.pop(0)
        time.sleep(0.05)
        return b""


def find_serial_port() -> str:
    candidates = sorted(glob.glob("/dev/serial/by-id/*")) + \
        sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    if not candidates:
        raise ControllerError("No USB serial device found (is the Arduino plugged in?). "
                              "Use --port /dev/ttyXXX or --port sim.")
    return candidates[0]
