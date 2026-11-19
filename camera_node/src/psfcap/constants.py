"""Fixed rig and capture settings. Edit here if the hardware changes."""

from pathlib import Path

# Sensor (OV9281 mono)
SENSOR_SIZE = (1280, 800)      # full resolution (w, h)
SENSOR_MAX = 1023              # 10-bit full scale; counted as saturated

# Stage (must match controller/include/PinMap.h)
BAUD = 115200
ZERO_DISTANCE_MM = 38.0        # source-to-sensor distance at ZERO
MAX_TRAVEL_MM = 320.0          # physical travel away from ZERO (software limit)
STEPS_PER_MM = 800.0           # STEPS_PER_MM in the firmware
AWAY_SIGN = -1                 # sign of the MOVE argument that moves the source away
MIN_SPEED_MM_S = 0.5           # conservative; only used to size serial timeouts

# Capture
SETTLE_S = 1.0                 # wait after each move for vibration to die down
GAIN = 1.0                     # analogue gain (1.0 = lowest noise)
INITIAL_EXPOSURE_US = 5000
MIN_EXPOSURE_US = 20
MAX_EXPOSURE_US = 500_000
TARGET_PEAK = 0.75             # auto exposure target, fraction of full scale
MIN_SNR = 50.0                 # warn below this peak SNR
DARK_FRAMES = 30

# Output / web
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "camera_node" / "data"
HTTP_PORT = 8002
