"""Run state shared between the sweep and the web UI."""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field, fields


@dataclass
class RunState:
    phase: str = "starting"
    message: str = ""
    index: int = 0
    total: int = 0
    target_distance_mm: float | None = None
    distance_mm: float | None = None
    exposure_us: int | None = None
    run_dir: str | None = None
    last_capture: dict | None = None
    warnings: list = field(default_factory=list)
    paused: bool = False
    started: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    last_jpeg: bytes | None = field(default=None, repr=False)
    pause_evt: threading.Event = field(default_factory=threading.Event, repr=False)
    abort_evt: threading.Event = field(default_factory=threading.Event, repr=False)

    def __post_init__(self):
        self.pause_evt.set()  # set = running

    def public(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)
             if f.name not in ("last_jpeg", "pause_evt", "abort_evt")}
        d["warnings"] = list(self.warnings)
        d["aborted"] = self.abort_evt.is_set()
        return d


class Aborted(Exception):
    pass
