"""Z sampling plans: which z positions a session visits, and named plans saved for reuse.

A plan is {name, z_min, z_max, n_positions, spacing} with spacing "inverse" (even
in 1/z, roughly even steps in PSF scale since magnification ~ 1 + d/z) or "linear".
Sessions at different (x, y) must share a plan to form a grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import coords
from .config import MIN_SPACING_STEPS, STEPS_PER_MM, Z_HOME_MM, Z_MAX_MM

PLAN_KEYS = ("z_min", "z_max", "n_positions", "spacing")


def validate(plan: dict) -> dict:
    z_min, z_max, n = float(plan["z_min"]), float(plan["z_max"]), int(plan["n_positions"])
    spacing = plan.get("spacing", "inverse")
    if not (coords.in_range(z_min) and coords.in_range(z_max)):
        raise ValueError(f"z must be within {Z_HOME_MM}..{Z_MAX_MM} mm")
    if z_max < z_min or (n > 1 and z_max == z_min):
        raise ValueError("z max must be larger than z min")
    if n < 1:
        raise ValueError("Number of positions must be at least 1")
    if spacing not in ("inverse", "linear"):
        raise ValueError("spacing must be 'inverse' or 'linear'")
    return {"name": plan.get("name") or None, "z_min": z_min, "z_max": z_max,
            "n_positions": n, "spacing": spacing}


def positions(plan: dict) -> dict:
    """Planned z, actual z after rounding to motor steps, and warnings."""
    plan = validate(plan)
    n = plan["n_positions"]
    if n == 1:
        planned = np.array([plan["z_min"]])
    elif plan["spacing"] == "linear":
        planned = np.linspace(plan["z_min"], plan["z_max"], n)
    else:
        planned = 1 / np.linspace(1 / plan["z_min"], 1 / plan["z_max"], n)
        planned[0], planned[-1] = plan["z_min"], plan["z_max"]     # exact ends
    actual = [coords.round_z(z) for z in planned]
    warnings = []
    if n > 1:
        gaps = np.diff(actual) * STEPS_PER_MM
        smallest = float(gaps.min())
        if smallest < MIN_SPACING_STEPS:
            warnings.append(f"Smallest spacing is {smallest:.0f} motor steps "
                            f"({smallest / STEPS_PER_MM * 1000:.1f} um); "
                            f"positions closer than {MIN_SPACING_STEPS} steps are not meaningful")
    return {**plan, "planned_z": [round(float(z), 6) for z in planned], "actual_z": actual,
            "warnings": warnings}


class PlanStore:
    """Named plans in data/plans.json."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "plans.json"

    def all(self) -> dict[str, dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self, plan: dict) -> dict:
        plan = validate(plan)
        if not plan["name"]:
            raise ValueError("A plan needs a name")
        plans = self.all()
        plans[plan["name"]] = plan
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(plans, indent=2))
        return plan

    def delete(self, name: str) -> None:
        plans = self.all()
        plans.pop(name, None)
        self.path.write_text(json.dumps(plans, indent=2))

    def matching_name(self, plan: dict) -> str | None:
        """Name of the saved plan with exactly these parameters, if any."""
        plan = validate(plan)
        for name, saved in self.all().items():
            if all(saved[k] == plan[k] for k in PLAN_KEYS):
                return name
        return None
