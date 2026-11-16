"""Run state shared between the sweep and the web UI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunState:
    phase: str = "starting"
    index: int = 0
    total: int = 0
    distance_mm: float | None = None
    exposure_us: int | None = None
    run_dir: str | None = None
    last_capture: dict | None = None
    warnings: list = field(default_factory=list)
    last_jpeg: bytes | None = field(default=None, repr=False)

    def public(self) -> dict:
        return {"phase": self.phase, "index": self.index, "total": self.total,
                "distance_mm": self.distance_mm, "exposure_us": self.exposure_us,
                "run_dir": self.run_dir, "last_capture": self.last_capture,
                "warnings": list(self.warnings)}
