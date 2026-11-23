"""Backend: owns the hardware and the application state, and enforces locking.

States: idle, capturing, waiting_for_user, error. Anything other than idle rejects
manual motor / exposure commands and model building, so a second tab or a reload
can never interfere with a running capture. `snapshot()` is what the frontend sees;
every change bumps `version`, which the server pushes as a server-sent event.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .config import DEFAULTS, Z_HOME_MM
from .plans import PlanStore

log = logging.getLogger(__name__)

IDLE, CAPTURING, WAITING, ERROR = "idle", "capturing", "waiting_for_user", "error"


class Busy(RuntimeError):
    """Command refused because of the current state (HTTP 409)."""


class Aborted(Exception):
    pass


class Backend:
    def __init__(self, camera, stage, data_dir: Path, world=None):
        self.camera, self.stage, self.world = camera, stage, world
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plans = PlanStore(self.data_dir)
        self.live = {"exposure_us": camera.exposure_us, "gain": camera.gain}

        self._cond = threading.Condition(threading.RLock())
        self.version = 0
        self.state = IDLE
        self.paused = False
        self.prompt: dict | None = None
        self.error: str | None = None
        self.progress: dict = {}
        self.warnings: list[str] = []
        self.session_id: str | None = None
        self.manual_busy: str | None = None      # e.g. "homing", while a manual command runs
        self.job: dict = {"running": False}      # model building progress
        self._abort = threading.Event()
        self._continue = threading.Event()
        self._thread: threading.Thread | None = None

    # --- status ---------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._cond:
            return {
                "version": self.version, "state": self.state, "paused": self.paused,
                "prompt": self.prompt, "error": self.error, "progress": self.progress,
                "warnings": self.warnings[-50:], "session_id": self.session_id,
                "z_mm": self.stage.z, "homed": self.stage.steps is not None,
                "manual_busy": self.manual_busy, "live": self.live, "job": self.job,
                "fake": [name for name, fake in (("camera", self.world is not None),
                                                 ("stage", type(self.stage.port).__name__ == "FakeArduino"))
                         if fake],
            }

    def changed(self, **fields) -> None:
        with self._cond:
            for k, v in fields.items():
                setattr(self, k, v)
            self.version += 1
            self._cond.notify_all()

    def wait_change(self, version: int, timeout: float) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self.version != version, timeout)

    def warn(self, message: str) -> None:
        log.warning(message)
        with self._cond:
            self.warnings.append(message)
        self.changed()

    # --- manual commands (Live tab) --------------------------------------------
    def _manual(self, what: str, fn):
        with self._cond:
            if self.state != IDLE:
                raise Busy(f"Not allowed while {self.state.replace('_', ' ')}")
            if self.manual_busy or self.job["running"]:
                raise Busy(f"Busy: {self.manual_busy or 'building model'}")
            self.manual_busy = what
        self.changed()
        try:
            return fn()
        finally:
            self.changed(manual_busy=None)

    def home(self) -> None:
        self._manual("homing", self.stage.home)

    def move(self, z_mm: float) -> float:
        return self._manual("moving", lambda: self.stage.move_to_z(z_mm))

    def set_live_controls(self, exposure_us: float, gain: float) -> dict:
        def apply():
            self.camera.set_controls(exposure_us, gain)
            self.live = {"exposure_us": self.camera.exposure_us, "gain": self.camera.gain}
            return self.live
        return self._manual("setting exposure", apply)

    # --- capture session ---------------------------------------------------------
    def start_session(self, params: dict) -> str:
        from .session import CaptureSession
        with self._cond:
            if self.state != IDLE or self.manual_busy or self.job["running"]:
                raise Busy("A session, manual command or model build is already running")
            session = CaptureSession(self, params)    # validates; raises ValueError
            self._abort.clear()
            self._continue.clear()
            self.state, self.paused, self.prompt, self.error = CAPTURING, False, None, None
            self.warnings = list(session.plan_warnings)
            self.session_id = session.id
            self.progress = {}
            self._thread = threading.Thread(target=self._run, args=(session,), name="capture", daemon=True)
            self._thread.start()
        self.changed()
        return session.id

    def _run(self, session) -> None:
        try:
            session.run()
            self.changed(state=IDLE, paused=False, prompt=None)
        except Aborted:
            session.finish("aborted")
            self.warn("Session aborted")
            self.changed(state=IDLE, paused=False, prompt=None)
        except Exception as e:                          # hardware or I/O failure
            log.exception("capture failed")
            session.finish("error", error=repr(e))
            self.changed(state=ERROR, paused=False, prompt=None, error=f"{type(e).__name__}: {e}")
        finally:
            try:
                self.camera.set_controls(self.live["exposure_us"], self.live["gain"])
            except Exception:
                log.exception("could not restore live controls")

    def pause(self, paused: bool) -> None:
        with self._cond:
            if self.state not in (CAPTURING, WAITING):
                raise Busy("No session running")
        self.changed(paused=paused)

    def abort(self) -> None:
        with self._cond:
            if self.state not in (CAPTURING, WAITING):
                raise Busy("No session running")
        self._abort.set()
        self._continue.set()
        self.changed(paused=False)

    def continue_prompt(self) -> None:
        with self._cond:
            if self.state != WAITING:
                raise Busy("Nothing to continue")
            if self.world is not None:                  # the fake light follows the prompts
                self.world.light_on = self.prompt["id"] == "light_on"
            self._continue.set()

    def clear_error(self) -> None:
        with self._cond:
            if self.state != ERROR:
                raise Busy("No error to clear")
        self.changed(state=IDLE, error=None)

    # --- used by the capture thread ---------------------------------------------
    def checkpoint(self) -> None:
        """Called between frames: raise on abort, block while paused."""
        while True:
            if self._abort.is_set():
                raise Aborted()
            with self._cond:
                if not self.paused:
                    return
                self._cond.wait(0.2)

    def ask_user(self, prompt_id: str, message: str) -> None:
        self._continue.clear()
        self.changed(state=WAITING, prompt={"id": prompt_id, "message": message})
        self._continue.wait()
        if self._abort.is_set():
            raise Aborted()
        self.changed(state=CAPTURING, prompt=None)

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.checkpoint()
            time.sleep(min(0.05, max(0.0, end - time.monotonic())))

    # --- model building (Model tab) ------------------------------------------------
    def require_idle_for_model(self) -> None:
        with self._cond:
            if self.state != IDLE:
                raise Busy("The Model tab is locked while a measurement is in progress")

    def run_job(self, fn) -> None:
        """Run fn(progress_callback) in the background, reporting into `job`."""
        with self._cond:
            self.require_idle_for_model()
            if self.job["running"] or self.manual_busy:
                raise Busy("Busy")
            self.job = {"running": True, "progress": 0.0, "message": "starting", "error": None}
        self.changed()

        def report(fraction: float, message: str):
            self.changed(job={**self.job, "progress": fraction, "message": message})

        def work():
            try:
                result = fn(report)
                self.changed(job={"running": False, "progress": 1.0, "message": "done",
                                  "error": None, "result": result})
            except Exception as e:
                log.exception("job failed")
                self.changed(job={"running": False, "progress": 0.0, "message": "failed",
                                  "error": f"{type(e).__name__}: {e}"})
        threading.Thread(target=work, name="job", daemon=True).start()

    def default_options(self) -> dict:
        return {**DEFAULTS, "max_exposure_us": self.camera.exposure_limits[1],
                "exposure_us": self.live["exposure_us"], "gain": self.live["gain"],
                "z_home": Z_HOME_MM}
