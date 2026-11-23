"""Session folders on disk.

  camera_node/data/<YYYY-MM-DDTHH-MM-SS>_x+5.0_y-2.5/
      metadata.json            updated after every frame (atomic replace)
      z012_p1_f03.tiff         raw PSF frame: position index, pass (1-based), frame (0-based)
      bg_e012000us_f00.tiff    background frame: requested exposure, frame
      processed/<key>/         cached processing results (see processing.py)

Frames are 16-bit TIFFs holding the raw 10-bit values unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from . import __version__

METADATA = "metadata.json"


def session_name(start: datetime, x_mm: float, y_mm: float) -> str:
    return f"{start:%Y-%m-%dT%H-%M-%S}_x{x_mm:+.1f}_y{y_mm:+.1f}"


def frame_name(z_index: int, pass_no: int, frame: int) -> str:
    return f"z{z_index:03d}_p{pass_no}_f{frame:02d}.tiff"


def background_name(exposure_us: int, frame: int) -> str:
    return f"bg_e{exposure_us:06d}us_f{frame:02d}.tiff"


def write_frame(path: Path, raw: np.ndarray) -> None:
    tifffile.imwrite(path, raw.astype(np.uint16, copy=False))


def read_frame(path: Path) -> np.ndarray:
    return tifffile.imread(path)


def write_json(path: Path, data: dict) -> None:
    """Write via a temporary file so a crash never leaves half a file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def load_metadata(session_dir: Path) -> dict:
    return json.loads((Path(session_dir) / METADATA).read_text())


def app_info() -> dict:
    try:
        repo = Path(__file__).parent
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                                    text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", False
    return {"name": "psf_capture", "version": __version__, "git_commit": commit or "unknown",
            "git_dirty": dirty}


def list_sessions(data_dir: Path) -> list[dict]:
    """Summaries of all sessions, newest first."""
    out = []
    for meta_path in sorted(Path(data_dir).glob(f"*/{METADATA}"), reverse=True):
        try:
            m = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "id": meta_path.parent.name,
            "status": m.get("status"),
            "x_mm": m.get("x_mm"), "y_mm": m.get("y_mm"),
            "plan_name": m["plan"].get("name"),
            "z_min": m["plan"]["z_min"], "z_max": m["plan"]["z_max"],
            "n_positions": m["plan"]["n_positions"], "spacing": m["plan"]["spacing"],
            "passes": m["options"]["passes"],
            "frames": len(m.get("frames", [])),
            "started": m.get("started"),
        })
    return out
