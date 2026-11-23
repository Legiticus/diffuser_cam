"""Fixed rig constants and defaults. Edit here if the hardware changes.

Contents:
  - paths and server port
  - stage (must match controller/include/PinMap.h)
  - sensor (OV9281)
  - optics used by the PSF model
  - capture defaults (the Capture tab's default values)
"""

from pathlib import Path

# --- Paths / server ---------------------------------------------------------
CAMERA_NODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CAMERA_NODE_DIR / "data"
HTTP_PORT = 8003

# --- Stage ------------------------------------------------------------------
SERIAL_PORT = "/dev/ttyUSB0"
BAUD = 115200
STEPS_PER_MM = 800            # firmware: (400 * 16 steps/rev) / 8 mm/rev
Z_HOME_MM = 10.0              # source-to-diffuser distance when the front limit switch zeros
Z_MAX_MM = 340.0              # software far travel limit (there is no back switch)
BACKLASH_MM = 1.0             # positions are approached from below z by at least this much
MIN_SPACING_STEPS = 4         # warn if planned positions are closer than this
HOME_TOLERANCE_MM = 0.05      # switch trigger moved more than this since the session start => steps lost
STAGE_SPEED_MM_S = 5.0        # measured travel speed (firmware loop limits it below its 16 mm/s setting)

# --- Sensor -----------------------------------------------------------------
SENSOR_MODEL = "OV9281"
SENSOR_SIZE = (1280, 800)     # (width, height)
BIT_DEPTH = 10
SATURATION = 1023             # raw value counted as clipped
PIXEL_PITCH_MM = 0.003        # 3 um pixels
PREVIEW_SIZE = (640, 400)     # MJPEG live view; never saved as data

# --- Optics (PSF model) -----------------------------------------------------
DIFFUSER_SENSOR_MM = 2.3      # d: used only when the shift per mm cannot be fitted
# Image direction of the PSF shift for a +x / +y source move (+x right, +y up as
# seen from the camera). Like a pinhole, the pattern moves opposite to the source;
# image rows grow downward. A negative fitted shift per mm means these are flipped.
SHIFT_SIGN_COL = -1
SHIFT_SIGN_ROW = +1

# --- Capture defaults -------------------------------------------------------
DEFAULTS = {
    "z_min": Z_HOME_MM,
    "z_max": Z_MAX_MM,
    "n_positions": 64,
    "spacing": "inverse",           # "inverse" (even in 1/z) or "linear"
    "frames_per_position": 5,
    "passes": 1,
    "background": True,
    "background_frames": None,       # None => same as frames_per_position
    "exposure_mode": "per_position",  # or "fixed"
    "target_fraction": 0.75,         # brightest pixel target, fraction of (saturation - black)
    "max_exposure_us": None,         # None => camera hardware limit
    "min_signal_fraction": 0.05,     # below this at max exposure => 'no PSF'
    "allowed_saturated_px": 0,
    "settle_s": 0.5,
    "exposure_us": 10000,            # Live tab start values
    "gain": 1.0,
}
# Per-position exposures are snapped to this many steps per doubling, which keeps
# the number of distinct exposures (each needs its own background) small.
EXPOSURE_STEPS_PER_OCTAVE = 2
