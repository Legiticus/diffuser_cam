#!/usr/bin/env python3
"""
DiffuserCam PSF depth-sweep capture.

Moves the point light source to a series of distances from the sensor using the
Arduino stepper controller (see ../../controller) and captures a calibration
point-spread function (PSF) from the OV9281 mono sensor at each one.

  * Raw, linear 10-bit sensor values, averaged over N frames, saved as float32 .npy.
  * Per-distance auto exposure: brightest pixels ~75% of full scale, never saturated.
    The actual exposure is recorded so PSFs can be normalised against each other.
  * Only frames whose exposure started after the stage stopped and settled are used.
  * Distances are uniform in 1/z (the PSF scales with 1 + t/z), snapped to whole
    motor steps, always approached moving outward (consistent backlash).
  * Travel is limited in software to 320 mm from zero (38..358 mm source to sensor).
    The back limit switch is not relied on; if one is fitted and trips, the run stops.
  * Dark/background frames at every exposure used (light source off, prompted at the end).
  * manifest.json records distances, exposures, statistics, warnings, and a fit of
    PSF size against 1/z that checks the stage really moved as commanded.

Fixed rig settings (38 mm zero distance, 800 steps/mm, travel direction, exposure
limits, ...) live in psfcap/constants.py.

Live debug view while it runs: http://<pi>:8002/
  GET /                       live view + status page
  GET /stream.mjpg            live MJPEG (contrast stretched, saturated pixels red)
  GET /api/status             JSON run state and last PSF stats
  GET /api/last_capture.jpg   last saved PSF

Output: camera_node/data/psf_YYYYMMDD_HHMMSS/
  psf_000_z0038.000mm.npy ...   PSFs (raw DN, not dark-subtracted)
  dark_exp00000320us.npy ...    dark frames, linked from each PSF's dark_file
  manifest.json

Example:
  python3 psf_capture.py --stop 300 --num 20
"""

import sys

from psfcap.cli import main

if __name__ == "__main__":
    sys.exit(main())
