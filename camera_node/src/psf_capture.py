#!/usr/bin/env python3
"""
DiffuserCam PSF depth-sweep capture.

Moves the point light source to a series of distances from the sensor using the
Arduino stepper controller (see ../../controller), and at every distance captures
a calibration point-spread function (PSF) from the OV9281 mono sensor.

What makes the data usable for DiffuserCam reconstruction:
  * Raw, linear 10-bit sensor values (R10 raw stream, no ISP, no gamma, no
    auto exposure / auto gain). Saved as float32 .npy, averaged over N frames.
  * Each PSF is checked for saturation (clipped PSFs ruin deconvolution) and for
    enough signal. Exposure is auto-bracketed per distance (or fixed) and the
    actual exposure from the sensor metadata is recorded so PSFs can be
    normalised against each other.
  * Only frames whose exposure started after the stage stopped and settled, and
    after the requested exposure took effect, are used.
  * Distances default to uniform spacing in 1/z. The PSF of a lensless camera
    scales with magnification (1 + t/z), so equal 1/z steps give equal changes
    in the PSF; linear spacing over-samples far depths.
  * Travel is limited in software to 320 mm from zero (38..358 mm source to
    sensor). There is no back limit switch on the rig, so the script never relies
    on one; if one is added later and reports "Back Limit", the run stops safely.
  * Stage moves are snapped to whole motor steps (800 steps/mm by default) and
    always go outward from the zero position, so lead-screw backlash stays on
    the same side.
  * Optional dark/background frames at every exposure used (light source off).
  * manifest.json / manifest.csv record distances, exposures, statistics and
    warnings. The PSF size is fitted against 1/z to check the stage really moved
    the distances requested.

Controller protocol (firmware in controller/src/main.cpp, 115200 baud):
  "ZERO\\n"       -> "ZERO_SUCCESS"   (homes to the front limit = zero_distance mm)
  "MOVE <mm>\\n"  -> "MOVE_SUCCESS" | "MOVE_FAIL - ..."   (relative move;
                    the firmware moves STEPPER_FORWARD*STEPS_PER_MM*mm steps, and
                    FORWARD points toward the front limit, so moving the light
                    AWAY from the sensor needs a negative mm value.)

Debug web UI / API (Flask, default http://<pi>:8002/):
  GET  /                    live view + status page
  GET  /stream.mjpg         live MJPEG (?stretch=0|1 &scale=0.5)
  GET  /api/status          JSON run state, current distance, last PSF stats
  GET  /api/latest.npy      latest raw frame (uint16, 10-bit values)
  GET  /api/last_capture.jpg   last saved PSF (stretched)
  POST /api/pause | /api/resume | /api/abort
  POST /api/zero, /api/goto?distance_mm=X   (only in --preview mode)

Examples:
  # 20 PSFs from 38 mm to 300 mm (max is 358 mm = 38 + 320 mm travel), uniform in 1/z, auto exposure, dark frames at end
  python3 psf_capture.py --stop 300 --num 20

  # explicit distances, fixed exposure, no dark frames
  python3 psf_capture.py --distances 38,50,75,100,150 --exposure-us 8000 --no-dark

  # camera + web viewer only (for aligning the point source)
  python3 psf_capture.py --preview

  # dry run without the Arduino attached
  python3 psf_capture.py --port sim --stop 120 --num 5 --no-dark --yes
"""

import sys

from psfcap.cli import main

if __name__ == "__main__":
    sys.exit(main())
