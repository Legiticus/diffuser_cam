"""Arduino stepper stage over USB serial (ZERO / MOVE protocol)."""

from __future__ import annotations

import glob
import logging
import math
import time

from .constants import AWAY_SIGN, BAUD, MAX_TRAVEL_MM, MIN_SPEED_MM_S, STEPS_PER_MM

log = logging.getLogger(__name__)


class ControllerError(RuntimeError):
    pass


class LimitHit(ControllerError):
    def __init__(self, which: str, msg: str):
        super().__init__(msg)
        self.which = which  # "front" | "back"


class StageController:
    """Tracks the light-source position in whole motor steps from zero."""

    def __init__(self, port: str):
        self.port = port
        # Software travel limit. This is the protection against running off the
        # back of the rail; the back limit switch may not exist, so it's never relied on.
        self.max_steps = int(math.floor(MAX_TRAVEL_MM * STEPS_PER_MM + 1e-6))
        self.position_steps: int | None = None   # None = unknown, needs ZERO
        self._ser = None

    def open(self):
        import serial  # pyserial
        log.info("Opening controller on %s @ %d", self.port, BAUD)
        self._ser = serial.Serial(self.port, BAUD, timeout=0.2)
        # Opening the port toggles DTR, which resets the Nano. Wait for the
        # bootloader to finish, then drop anything it printed.
        time.sleep(2.5)
        self._ser.reset_input_buffer()

    def close(self):
        if self._ser:
            self._ser.close()
            self._ser = None

    def _command(self, cmd: str, ok: str, timeout: float) -> str:
        assert self._ser is not None
        self._ser.reset_input_buffer()
        log.debug("-> %s", cmd)
        self._ser.write((cmd + "\n").encode("ascii"))
        self._ser.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._ser.readline().decode("ascii", errors="replace").strip()
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

    def zero(self):
        self.position_steps = None
        self._command("ZERO", "ZERO_SUCCESS", 30.0 + MAX_TRAVEL_MM / MIN_SPEED_MM_S)
        self.position_steps = 0

    @staticmethod
    def mm_to_steps(travel_mm: float) -> int:
        return int(round(travel_mm * STEPS_PER_MM))

    def move_to_steps(self, target_steps: int):
        """Absolute move (in steps away from zero) with exact step bookkeeping."""
        if self.position_steps is None:
            raise ControllerError("Position unknown: run ZERO first")
        if not 0 <= target_steps <= self.max_steps:
            raise ControllerError(
                f"Target {target_steps / STEPS_PER_MM:.3f} mm travel is outside the "
                f"allowed range 0..{self.max_steps / STEPS_PER_MM:.3f} mm; move refused")
        delta = target_steps - self.position_steps
        if delta == 0:
            return
        fw_steps = AWAY_SIGN * delta          # steps in firmware's frame
        # Firmware does long(800.0f * atof(arg)) which truncates toward zero.
        # Add half a step so float rounding still lands on the exact count.
        arg = (fw_steps + math.copysign(0.5, fw_steps)) / STEPS_PER_MM
        timeout = 15.0 + abs(delta) / STEPS_PER_MM / MIN_SPEED_MM_S
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
        return self.position_steps / STEPS_PER_MM


def find_serial_port() -> str:
    candidates = sorted(glob.glob("/dev/serial/by-id/*")) + \
        sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    if not candidates:
        raise ControllerError("No USB serial device found (is the Arduino plugged in?). "
                              "Use --port /dev/ttyXXX.")
    return candidates[0]
