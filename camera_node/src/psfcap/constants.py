"""Hardware and path constants shared across the package."""

from __future__ import annotations

from pathlib import Path


SENSOR_SIZE = (1280, 800)      # OV9281 full resolution (w, h)
SENSOR_MAX = 1023              # 10-bit full scale
MAX_TRAVEL_MM = 320.0          # physical travel away from the ZERO position
REPO_ROOT = Path(__file__).resolve().parents[3]
