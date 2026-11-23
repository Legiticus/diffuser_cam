"""Motorised z stage: the Arduino serial protocol (see controller/src/main.cpp).

  Stage        - backend view of the slider; tracks the position in whole steps,
                 enforces the travel limits and the approach direction.
  open_serial  - opens the real USB serial port.
  FakeArduino  - in-memory stand-in for the serial port that emulates the firmware,
                 so the full flow runs on a PC.

Firmware protocol (one command per line, 115200 baud):
  ZERO        -> "ZERO_SUCCESS", "POSITION <mm>"
  MOVE <mm>   -> "MOVE_SUCCESS" or "MOVE_FAIL - <reason>", then "POSITION <mm>"
MOVE is relative, positive towards the camera, and the firmware truncates
800 * mm to whole steps, so arguments are sent with a half-step bias.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np

from . import coords
from .config import (BACKLASH_MM, BAUD, HOME_TOLERANCE_MM, STAGE_SPEED_MM_S,
                     STEPS_PER_MM, Z_HOME_MM, Z_MAX_MM)

log = logging.getLogger(__name__)


class StageError(RuntimeError):
    pass


class LimitHit(StageError):
    """The front limit switch triggered during a move at `slider_mm`."""

    def __init__(self, slider_mm: float):
        super().__init__(f"Front limit switch triggered at {slider_mm:.3f} mm")
        self.slider_mm = slider_mm


class Stage:
    def __init__(self, port):
        self.port = port
        self.steps: int | None = None          # None until homed
        self.max_steps = coords.z_to_steps(Z_MAX_MM)
        self.backlash_steps = round(BACKLASH_MM * STEPS_PER_MM)
        self._lock = threading.Lock()

    # --- public -------------------------------------------------------------
    @property
    def z(self) -> float | None:
        return None if self.steps is None else coords.steps_to_z(self.steps)

    def home(self) -> None:
        with self._lock:
            self._home()

    def move_to_z(self, z_mm: float) -> float:
        """Move to z, always arriving from below (away from the camera).
        Returns the z actually reached (rounded to whole steps)."""
        with self._lock:
            if self.steps is None:
                raise StageError("Position unknown: home the stage first")
            if not coords.in_range(z_mm):
                raise StageError(f"z={z_mm} mm is outside {Z_HOME_MM}..{Z_MAX_MM} mm")
            target = coords.z_to_steps(z_mm)
            if target < self.steps:                # coming down: overshoot, then go up
                pre = target - self.backlash_steps
                if pre < 0:
                    self._home()                   # homing also ends moving away from the camera
                else:
                    self._move_steps(pre)
            if target != self.steps:
                self._move_steps(target)
            return self.z

    def check_home(self) -> dict:
        """Drive slowly into the front switch, record where it triggers (slider mm,
        ideally ~0) and re-home. A trigger far from 0 means steps were lost."""
        with self._lock:
            if self.steps is None:
                raise StageError("Position unknown: home the stage first")
            trigger_mm = None
            try:
                if self.steps != self.backlash_steps:
                    self._move_steps(self.backlash_steps)
                self._move_steps(-10 * STEPS_PER_MM, allow_negative=True)   # up to 10 mm past home
            except LimitHit as e:
                trigger_mm = e.slider_mm
            self._home()
            lost = trigger_mm is None or abs(trigger_mm) > HOME_TOLERANCE_MM
            return {"trigger_position_mm": trigger_mm, "tolerance_mm": HOME_TOLERANCE_MM,
                    "steps_lost": lost}

    def close(self) -> None:
        self.port.close()

    # --- protocol -----------------------------------------------------------
    def _home(self) -> None:
        self.steps = None
        status, pos = self._command("ZERO", timeout=30 + 2 * Z_MAX_MM / STAGE_SPEED_MM_S)
        if status != "ZERO_SUCCESS" or pos is None or coords.slider_mm_to_steps(pos) != 0:
            raise StageError(f"Homing failed: {status} / position {pos}")
        self.steps = 0

    def _move_steps(self, target: int, allow_negative: bool = False) -> None:
        if not allow_negative and not 0 <= target <= self.max_steps:
            raise StageError(f"Target {target} steps outside 0..{self.max_steps}")
        delta = target - self.steps
        if delta == 0:
            return
        # Firmware: steps = trunc(800.0 * arg). Bias by half a step so float error
        # can never round it down by one.
        arg = coords.steps_to_slider_mm(delta + (0.5 if delta > 0 else -0.5))
        timeout = 10 + 2 * abs(delta) / STEPS_PER_MM / STAGE_SPEED_MM_S
        status, pos = self._command(f"MOVE {arg:.6f}", timeout)
        self.steps = None if pos is None else coords.slider_mm_to_steps(pos)
        if status.startswith("MOVE_FAIL") and "Front Limit" in status and pos is not None:
            raise LimitHit(pos)
        if status != "MOVE_SUCCESS":
            raise StageError(f"{status} (position {pos})")
        if self.steps != target:
            raise StageError(f"Stage reports {self.steps} steps, expected {target}")

    def _command(self, cmd: str, timeout: float) -> tuple[str, float | None]:
        """Send a command; return (status line, reported slider position or None)."""
        self.port.reset_input_buffer()
        log.debug("-> %s", cmd)
        self.port.write((cmd + "\n").encode("ascii"))
        status = ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.port.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            log.debug("<- %s", line)
            if line.startswith("POSITION"):
                value = line.split(maxsplit=1)[1]
                return status, None if value == "UNKNOWN" else float(value)
            if line.startswith("ERROR"):
                raise StageError(f"Controller: {line}")
            status = line
        self.steps = None
        raise StageError(f"Timed out after {timeout:.0f} s waiting for a reply to '{cmd}'")


def open_serial(device: str):
    import serial  # pyserial
    port = serial.Serial(device, BAUD, timeout=0.5)
    time.sleep(2.5)            # opening the port resets the Nano; wait for its boot message
    port.reset_input_buffer()
    return port


class FakeArduino:
    """Emulates the firmware behind a serial-port-like interface.

    `true_steps` is the real distance from the switch (only the fake world knows it);
    `lose_steps` adds that many steps of error to every move to simulate stalls.
    `speed_scale` scales simulated travel time (0 = instant)."""

    HOME_BACKOFF = 8                        # ZERO backs off 0.01 mm steps until the switch releases

    def __init__(self, start_mm: float = 100.0, speed_scale: float = 0.0, lose_steps: int = 0):
        self.true_steps = round(start_mm * STEPS_PER_MM)
        self.counter = 0                    # firmware currentPosition(), +ve towards camera
        self.known = False
        self.speed_scale = speed_scale
        self.lose_steps = lose_steps
        self._out: deque[bytes] = deque()

    @property
    def z(self) -> float:
        return Z_HOME_MM + (self.true_steps - self.HOME_BACKOFF) / STEPS_PER_MM

    def write(self, data: bytes) -> None:
        for line in data.decode().splitlines():
            self._handle(line.strip())

    def readline(self) -> bytes:
        return self._out.popleft() if self._out else b""

    def reset_input_buffer(self) -> None:
        self._out.clear()

    def close(self) -> None:
        pass

    def _say(self, text: str) -> None:
        self._out.append((text + "\r\n").encode())

    def _report(self) -> None:
        self._say(f"POSITION {self.counter / STEPS_PER_MM:.3f}" if self.known else "POSITION UNKNOWN")

    def _travel(self, steps: int) -> None:
        if self.speed_scale:
            time.sleep(self.speed_scale * abs(steps) / STEPS_PER_MM / STAGE_SPEED_MM_S)

    def _handle(self, line: str) -> None:
        cmd, _, arg = line.partition(" ")
        if cmd == "ZERO":
            self._travel(self.true_steps)
            self.true_steps = self.HOME_BACKOFF
            self.counter = 0
            self.known = True
            self._say("ZERO_SUCCESS")
            self._report()
        elif cmd == "MOVE":
            if not arg:
                self._say("MOVE_FAIL - No argument passed")
                self._report()
                return
            steps = int(np.float32(STEPS_PER_MM) * np.float32(float(arg)))   # truncates like the AVR
            if steps > 0 and steps >= self.true_steps:                       # runs into the switch
                self._travel(self.true_steps)
                self.counter += self.true_steps
                self.true_steps = 0
                self._say("MOVE_FAIL - Front Limit Triggered")
            else:
                self._travel(steps)
                self.counter += steps
                self.true_steps -= steps + (self.lose_steps if steps < 0 else -self.lose_steps)
                self._say("MOVE_SUCCESS")
            self._report()
        else:
            self._say("ERROR - Not a valid command")
