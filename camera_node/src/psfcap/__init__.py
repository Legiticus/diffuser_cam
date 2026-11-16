"""DiffuserCam PSF depth-sweep capture package.

Modules:
  constants   hardware constants (sensor size, bit depth, max stage travel)
  controller  Arduino stage over serial (StageController) + SimulatedSerial
  camera      Picamera2 raw 10-bit capture thread (Camera, Frame)
  analysis    PSF statistics and 8-bit display conversion
  planning    distance plan (uniform in 1/z, linear, or explicit)
  state       RunState shared between the sweep and the web UI
  sweep       PsfSweep: move / settle / auto-expose / capture / save / validate
  web         Flask debug UI and API
  cli         argument parsing and main()

Run with `python3 psf_capture.py ...` or `python3 -m psfcap ...` from camera_node/src.
"""

from .constants import MAX_TRAVEL_MM, SENSOR_MAX, SENSOR_SIZE
from .controller import ControllerError, LimitHit, SimulatedSerial, StageController
from .cli import main

__all__ = ["MAX_TRAVEL_MM", "SENSOR_MAX", "SENSOR_SIZE", "ControllerError", "LimitHit",
           "SimulatedSerial", "StageController", "main"]
