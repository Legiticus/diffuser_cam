"""DiffuserCam PSF depth-sweep capture package.

Modules:
  constants   fixed rig and capture settings (edit here if the hardware changes)
  controller  Arduino stage over serial (StageController)
  camera      Picamera2 raw 10-bit capture thread (Camera, Frame)
  analysis    PSF statistics and display conversion
  state       RunState shared between the sweep and the web view
  sweep       distance plan and PsfSweep (move / settle / auto-expose / capture / save / validate)
  web         Flask debug view
  cli         argument parsing and main()

Run with `python3 psf_capture.py ...` or `python3 -m psfcap ...` from camera_node/src.
"""

from .cli import main

__all__ = ["main"]
