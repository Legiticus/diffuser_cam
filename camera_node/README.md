# PSF Capture

A web app for measuring the point spread function (PSF) of the diffuser camera.
It runs on the Raspberry Pi and drives two pieces of hardware: the OV9281 camera
and the Arduino z stage (USB serial). A point source moves along z under motor
control. You set the x/y offset of the source by hand and enter it for each
session. If you repeat sessions at several (x, y) offsets with the same z
sampling plan, you get a 3-D grid of measured PSFs, and the Model tab predicts
the PSF at any point from that grid.

## Architecture

```
 Browser (Alpine.js, no build)            Raspberry Pi
 ┌───────────────────────────┐   HTTP    ┌──────────────────────────────────────────────┐
 │ Live │ Capture │ Model     │ ───────▶ │ FastAPI on :8003 (server.py)                 │
 │                           │ ◀─────── │   REST API · SSE status · MJPEG · static     │
 └───────────────────────────┘ SSE/MJPEG│                    │                         │
                                         │              Backend (core.py)               │
                                         │   state: idle / capturing /                  │
                                         │          waiting_for_user / error            │
                                         │     │            │               │           │
                                         │ CaptureSession  processing.py   model.py     │
                                         │ (session.py)    (raw → PSFs)    (generator)  │
                                         │     │                                        │
                                         │ camera.py ── picamera2 / FakeCamera          │
                                         │ stage.py  ── USB serial / FakeArduino        │
                                         └──────────────────────────────────────────────┘
```

- **One server.** FastAPI serves the API and the static frontend from the same
  port, so no CORS setup is needed.
- **The backend owns the state.** While the state is anything other than `idle`,
  the backend refuses manual stage and exposure commands and model building
  (HTTP 409), whatever the UI shows. A second tab or a page reload therefore
  can't disturb a running capture. The frontend gets every state change over
  server-sent events and reconnects to a running session after a reload.
- **Capture and processing are separate.** A capture saves raw frames only. All
  corrections happen later in a repeatable processing step, which never
  modifies the raw files.
- **Fakes stand in for missing hardware.** The camera and the stage each have a
  real implementation and a fake one, so the whole app runs on a normal PC.

## File structure

```
camera_node/
├── README.md               this file
├── requirements.txt        pip packages (picamera2 and numpy come from apt on the Pi)
├── pytest.ini              test config; hardware tests are skipped by default
├── psf_capture/            the application (python -m psf_capture)
│   ├── __main__.py         command line: picks real/fake hardware, starts uvicorn
│   ├── config.py           rig constants and defaults (steps/mm, z range, sensor, optics)
│   ├── coords.py           the ONLY place z (mm) is converted to/from slider coordinates
│   ├── stage.py            Arduino protocol, travel limits, approach direction, homing check; FakeArduino
│   ├── camera.py           RealCamera (picamera2 raw stream), FakeCamera (synthetic caustic), frame stats
│   ├── plans.py            z sampling plans (linear / even in 1/z), rounding, named plan store
│   ├── storage.py          session folder layout, TIFF + metadata.json writing, session list
│   ├── core.py             Backend: state machine, locking, manual commands, pause/abort/prompts, jobs
│   ├── session.py          the capture procedure (homing, sweep, exposure control, backgrounds)
│   ├── processing.py       background subtraction, averaging, pass registration, normalisation, cache
│   ├── model.py            PSF generator (nearest measured PSF, shifted; points summed)
│   ├── server.py           HTTP routes, SSE, MJPEG, downloads
│   └── static/             frontend: index.html, app.js, style.css, vendor/alpine.min.js
├── tests/                  pytest suite (fake hardware + synthetic data)
│   ├── conftest.py         Backend fixture wired to the fakes
│   ├── synthetic.py        writes sessions from known synthetic PSFs
│   ├── test_stage.py       conversions, step rounding, limits, approach direction, lost steps
│   ├── test_session.py     full capture flow, exposure modes, no-PSF, locking, pause/abort
│   ├── test_processing.py  known-shift recovery, masks, background, normalisation, drift
│   ├── test_model.py       shift-per-mm fit, nearest plane, summing, range warnings
│   ├── test_api.py         HTTP API end to end, 409 locking, downloads, SSE
│   └── test_hardware.py    real camera / stage (pytest -m hardware)
└── data/                   collected sessions (git-ignored)
```

## Coordinate system

| Symbol | Meaning |
|---|---|
| **z** | Distance from the diffuser plane to the point source, in mm. 10 mm at home (front limit switch), 340 mm maximum (software limit). |
| **x** | Horizontal offset of the source from the optical axis, in mm. +x is to the right as seen from the camera looking at the source. |
| **y** | Vertical offset of the source from the optical axis, in mm. +y is up. |

- The Arduino zeros its position at the front limit switch, which is
  z = 10 mm. Its sign is the opposite of z: moving away from the camera is
  negative. `coords.py` is the only place that converts between the two:
  `slider_mm = -(z - 10)`.
- The firmware runs at 800 steps/mm (400 steps/rev × 16 microsteps over an
  8 mm lead), so one step is 1.25 µm. Every z position is rounded to whole
  steps, and the rounded ("actual") z is what gets recorded.
- There is no back switch. The backend homes at the start of every session and
  enforces z ≤ 340 mm in software.
- Every position is approached moving away from the camera, to take up
  backlash. To reach a lower z, the stage first goes 1 mm (`BACKLASH_MM`)
  below the target and then moves up. If that would go below 10 mm, it
  re-homes instead, because homing also finishes with a move away from the
  camera.
- **Homing check.** At the end of a session the stage moves to z = 11 mm and
  creeps into the switch. The slider position where the switch triggers is
  recorded. Ideally it is about 0; anything beyond ±0.05 mm
  (`HOME_TOLERANCE_MM`) is flagged as lost steps. The stage then re-homes.
- **Image directions.** A source move of +x shifts the PSF by
  `SHIFT_SIGN_COL · Δx · d/z` pixels along image columns, and +y shifts it by
  `SHIFT_SIGN_ROW · Δy · d/z` along image rows. The signs are set in
  `config.py`. They default to the pinhole-like inversion (−1 and +1), with
  d = 2.3 mm and 3 µm pixels. If the fitted shift per mm comes out negative,
  the Model tab warns that the signs are flipped.

## Data format

Each session gets its own folder, `camera_node/data/<start time>_x<±x>_y<±y>/`,
for example `2026-11-22T14-03-10_x+5.0_y-2.5/`:

| File | Content |
|---|---|
| `z012_p1_f03.tiff` | Raw frame: position index 12, pass 1, frame 3. 16-bit TIFF holding the raw 10-bit values unchanged (0–1023), full 1280×800. |
| `bg_e012000us_f00.tiff` | Background (light off) frame at a requested exposure of 12000 µs, frame 0. |
| `metadata.json` | Rewritten atomically after every frame, so a crash or abort keeps everything taken so far. |
| `processed/<key>/` | Processing cache: `psf.npy` (float32 `[n_z, 800, 1280]`, NaN = no data) and `info.json` (steps and parameters used, pass shifts, drift). |

`metadata.json` contains:

- `app`: name, version, git commit, and whether the working tree had local changes
- `status` (`running` / `complete` / `aborted` / `error`), plus `started` and
  `ended` timestamps
- `x_mm`, `y_mm`
- `plan`: name (if saved), `z_min`, `z_max`, `n_positions`, `spacing`,
  `planned_z` and `actual_z` (rounded to motor steps) for every position
- `options`: every measurement option used, with defaults filled in
- `camera`: sensor, size, `bit_depth` 10, `saturation_level` 1023,
  `black_level` (64 on the OV9281)
- `positions`: one entry per position and pass, with `z_mm`, requested exposure
  and gain, and `no_psf`
- `frames`: one entry per frame, with file, z index, pass, frame number, z,
  requested exposure, the **exposure and gain reported by the camera for that
  frame**, timestamp, saturated pixel count and peak value
- `backgrounds`: the same per-frame information for background frames
- `homing_checks` and `homing_check`: limit trigger position, tolerance,
  `steps_lost`
- `warnings`

**Z sampling plans** are stored by name in `data/plans.json`. A session records
the name of the plan it used. If you edit the fields so they match a saved plan
exactly, the session still records that plan's name. The Model tab warns when
the sessions you select used different plans.

## Capture procedure

1. Home the stage.
2. For each pass and each planned z: move there (approaching from below), wait
   for the settle time (default 0.5 s), set the exposure, and take the frames.
   Each frame is written to disk as soon as it arrives.
   - **Per-position exposure** (default). Gain stays at the Live tab value. The
     backend picks the longest exposure that keeps the brightest pixel at or
     below the target, which defaults to 75% of (saturation − black level).
     "Brightest pixel" means the 10th-brightest pixel, so a few hot pixels are
     ignored. Exposures are snapped to a ladder of 2 steps per doubling
     (`EXPOSURE_STEPS_PER_OCTAVE`) so that only a few distinct exposures need
     background frames. If any saved frame has saturated pixels, the
     position's frames are discarded and retaken one step lower. If even the
     maximum exposure gives a peak below the no-PSF threshold (default 5% of
     the range), the position is marked `no_psf`, a warning is raised, and the
     sweep moves on.
   - **Fixed exposure.** Every frame uses the Live tab exposure and gain.
     Saturated frames raise a warning.
   - The backend only accepts frames whose *reported* exposure and gain match
     the request and that started after the settle period. A control change
     takes about 7 frames to reach the OV9281, so for long exposures a change
     can take several seconds.
3. Homing check (see Coordinate system).
4. Background frames, if enabled. The app asks you to turn the light off, takes
   frames at every exposure used (by default as many as frames per position),
   then asks you to turn the light back on. A separate background is needed
   for each exposure because the black level is fixed while ambient light and
   dark current grow with exposure time.

**Pause** stops between frames. **Abort** stops after the current frame and
marks the session `aborted`. All frames taken so far are kept.

## Processing (Model tab)

You choose which steps run. The result is cached under a key made from the
steps and parameters, so changing a step reprocesses and repeating a choice
reuses the cache.

1. **Background subtraction.** Subtracts the mean background frame taken at
   the same exposure (`frames`), only the black level (`black_level`), or
   nothing. The result stays floating point and negative values are kept.
2. **Frame averaging.** Takes the mean of the frames at each position within a
   pass. When off, only the first frame is used.
3. **Pass registration and averaging.** The sharpest pass is the reference.
   Each pass is registered to it with subpixel phase correlation (Hann
   window, Gaussian-smoothed cross-power peak, log-parabola fit), then
   shifted onto the *mean* position of all passes. Shifting to the mean
   averages out random jitter but keeps any shift that is common to every
   pass. Passes whose sharpness or correlation is below `min_quality` × the
   median are excluded. Pixels with no data after shifting are NaN, and the
   combination (mean or median) ignores them. Passes are brought to a common
   exposure first.
   - The measured shifts appear as plots. Random scatter between passes
     means vibration.
   - **Drift with z.** The PSFs of neighbouring z planes are registered after
     rescaling each plane by the predicted magnification ratio (1 + d/z). The
     part of the drift that the nominal x/y does not explain is converted into
     a source x/y correction for each z, relative to the first plane, and a
     rail slope is fitted to it. On axis, a steady drift means the rail is
     tilted.
4. **Normalisation.** Divides by exposure × gain (counts per µs per unit
   gain, which keeps the real fall-off of brightness with distance), scales
   each plane to sum to 1, or leaves the data in counts at the first pass's
   exposure.

## PSF generator

The generator never blends neighbouring PSFs pixel by pixel, because blending
two shifted caustics gives two faint copies instead of one pattern in between.

- **Lateral.** It takes the PSF measured at the nearest (x, y) offset and shifts
  it by k(z) · Δ pixels. k(z) is the shift per mm at that z, fitted by least
  squares from how the PSF moves between sessions at different offsets. With
  only one offset, it uses d / (z · pixel pitch) with d = 2.3 mm.
- **Depth.** It uses the nearest measured z plane. If that plane is marked
  `no_psf`, it uses the next nearest and warns.
- **Several points.** Their PSFs are summed, weighted by intensity, which is
  valid for an incoherent source. Points outside the measured x, y or z range
  give warnings.
- **Downloads.** 16-bit PNG and TIFF, scaled so the maximum is 65535 (the
  scale factor is shown), and float32 `.npy`. In all outputs, negative values
  are clipped to 0 and pixels with no data are set to 0.

Points JSON (units are mm, `intensity` is optional and defaults to 1):

```json
{ "points": [ { "x": 0.0, "y": 0.0, "z": 120.0, "intensity": 1.0 } ] }
```

## Running on the Raspberry Pi

picamera2 is installed through apt, so the virtual environment must see the
system packages:

```bash
sudo apt install -y python3-picamera2 python3-numpy      # usually preinstalled on Raspberry Pi OS
cd ~/source/diffuser_cam/camera_node
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m psf_capture                           # real camera + Arduino on /dev/ttyUSB0
```

Then open `http://<pi-address>:8003` in a desktop browser.

Command-line options:

| Option | Effect |
|---|---|
| `--serial /dev/ttyACM0` | Use a different serial port. |
| `--fake-stage` | Real camera, simulated stage. Useful while the motor is unpowered. |
| `--fake-camera` | Simulated camera, real stage. |
| `--fake` | Simulate both. |
| `--fake-speed 0` | Make the fakes instant (1 = real time). |
| `--port`, `--data-dir`, `--debug` | Server port, data folder, debug logging. |

Only one process can use the camera, so stop other camera apps first. Opening
the serial port resets the Arduino; the app waits 2.5 s for it to boot.

## Running on a PC (no hardware)

```bash
cd camera_node
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt numpy
.venv/bin/python -m psf_capture --fake                     # http://localhost:8003
```

The fake camera renders a synthetic caustic PSF. It shifts with x/y (by
d/z), scales with z (by 1 + d/z), gets dimmer as 1/z², and jitters slightly
from frame to frame to imitate vibration. The fake light follows the
light-off and light-on prompts. The fake Arduino emulates the firmware
protocol, including step truncation and the limit switch.

## Tests

```bash
.venv/bin/python -m pytest              # everything except hardware (fakes + synthetic data)
.venv/bin/python -m pytest -m hardware  # on the Pi, app stopped: real camera and stage
```

The processing tests use synthetic data with known answers. For example, they
apply a known subpixel shift and check that it is recovered within 0.05 px, put
jitter on the passes and check that it is undone, and tilt the rail and check
that the slope is detected.
