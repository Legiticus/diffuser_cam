## Goal

Create a web-based application to facilitate the collection of data for the point spread function (PSF) of a diffuser-camera. Only the z (depth) position of the point source is motorized; the x and y offsets are set by hand and entered into the application for each session. By repeating sessions at different (x, y) offsets with the same z sampling plan, I will build a three dimensional grid of measured PSFs (each PSF being a 2D image) so that I can reliably approximate what the PSF is at any given point.

## Context

The PSF_Capture application is responsible for providing a simple and easy to use interface for facilitating the collection of PSF data. This application will be run on a raspberry pi connected via serial communication to an Arduino that controls the stepper motor and has a limit switch at the front only. For the code of the Arduino and how it communicates, see the `controller` folder. Do not modify the Arduino code.

### Coordinate System

- **z** is the distance from the diffuser plane to the point source, in mm. It is not the slider coordinate. The conversion between slider position (mm) and z (mm) must live in one place in the code. **z** ranges from 10 mm (home) to 340 mm (max). (Note that the Arduino has the opposite sign for its direction.)
- When the Arduino hits the frontal limit switch, it will zero its coordinate system. This represents a 10mm gap between the diffuser (which is ~2.3 mm away from the sensor) and the point light source.
- **x** (horizontal) and **y** (vertical) are the offsets of the point source from the optical axis, in mm.
- Sign convention: +x to the right and +y up, as seen from the camera looking at the source.

### Hardware Assumptions

- The point source should be small enough to be unresolved by the camera (pinhole diameter / z smaller than the camera's angular resolution at the closest z). Note that the pinhole diameter is 30 $\mu$m. About 2 pixels of blur at z = 10 mm, under 1 pixel beyond 20 mm; acceptable.
- The light cone of the source must cover the whole diffuser at the closest z and the largest x/y offset, otherwise the measured PSF is clipped.

## Project Location, GIT, & Standards

The code for this application will go in the camera_node folder and this application will be created on a branch called "psf_collector". The code should be simple, readable and minimal. Tests go in `camera_node/tests/` and collected data goes in `camera_node/data/`. The data directory must be excluded from git via `.gitignore`.

### Develop, Test, Deploy

- **Develop:** The backend must be able to run on a regular PC without the Pi hardware (Though it will run on the Pi hardware in action). The camera and the Arduino serial link are each accessed either with the real implementation or a fake (simulated) implementation for testing. The fake camera produces a synthetic PSF-like pattern that shifts and scales with the simulated source position, so the full capture flow and the UI can be exercised off-device.
- **Test:** Tests use pytest. Tests that need real hardware are marked (e.g. `@pytest.mark.hardware`) and skipped by default. Processing code (registration, background subtraction, normalization, PSF generation) is tested on synthetic data with known answers (e.g. apply a known shift and check it is recovered).
- **Deploy:** Document how to install and run on the Pi. picamera2 is installed via apt, so the Python virtual environment must be created with `--system-site-packages`.

## Architecture

The architecture should consist of two components. A JavaScript front end and a python backend. A single FastAPI server on port 8003 serves both the API and the static frontend files (Alpine.js needs no build step), which avoids cross-origin (CORS) configuration.

### Backend State

The backend owns the application state and enforces locking. It is a simple state machine: `idle`, `capturing`, `waiting_for_user` (e.g. "turn off the light"), `error`. While not `idle`, the backend rejects manual motor and exposure commands regardless of what the UI shows, so a second browser tab or a page reload cannot interfere with a running capture. The frontend reflects this state and reconnects to a running session after a reload. The backend will be running on the Pi during actual use of this hardware.

### Frontend

The JavaScript front end written with alpine.js (If another framework would be better for the task, ask) that has three tabs: **Live**, **Capture**, and **Model**.

- **Live:** A live video stream with manual controls.
  - Stepper motor movable up to a resolution of 0.1 mm, a current position readout (in z mm), and a Home button.
  - Manual exposure time and analog gain controls.
  - A histogram and a saturation indicator (percentage/count of clipped pixels).
- **Capture:** Collecting the data. When data collection is in process, the manual adjustments on the Live tab are locked. The Capture tab has these required inputs before a session can be run:
  - The minimum and maximum source distance z (bounded by the slider's physical range, see Coordinate System. Defaults are the min and max distances of 10 mm and 340 mm).
  - The x and y offsets of the point light source.

  Other options are optional and are described in more detail in the Measurement Options section below. The tab shows progress (current position, frame count, estimated time remaining), shows prompts when the backend is `waiting_for_user`, and has Pause and Abort buttons.
- **Model:** Also locked while measurement is in process (to keep the Pi's CPU free for capture). This tab takes collected data and builds a synthetic PSF generator from it.
  - The user selects which sessions to use and which processing steps to apply (background subtraction, normalization, pass registration/averaging — see Processing).
  - The user inputs a point in 3D space (x, y, z, intensity) and views what the PSF should be on the camera sensor. The result can be downloaded (16-bit PNG/TIFF and .npy).
  - The user can enter multiple points or upload a JSON file of points. The PSFs of multiple points are summed, which is valid for an incoherent source such as an LED. JSON format:

    ```json
    { "points": [ { "x": 0.0, "y": 0.0, "z": 120.0, "intensity": 1.0 } ] }
    ```

    Units are mm; `intensity` is optional and defaults to 1.0.
  - Points outside the measured range of x, y or z raise warnings rather than a silently extrapolated result.

#### Layout & Color Scheme

The design of the front end will include a dark mode display with red highlights on edges. The overall look should be a simple, modern, professional, and streamlined look designed for desktop screen use.

#### Technical Details

- The live view is an MJPEG stream over HTTP using a low-resolution preview stream. The preview is for viewing only and is never saved as data.
- Capture progress and state changes are pushed to the frontend with a WebSocket or server-sent events.

### Backend

The backend will be written in python using FastAPI and will interface with the camera. The camera is a `OV9281-110 1MP Mono Camera with Global Shutter More Clear Imaging` (1280×800, 10-bit). The device is a raspberry pi and the Arduino stepper controller is connected via serial with USB. The backend will need to support the features described in the frontend.

#### Camera

- Use picamera2. Auto exposure and auto gain are always disabled during a capture session; exposure time and analog gain are set explicitly.
- Data frames are taken from the raw 10-bit stream at full resolution and stored at 16 bits. They are never taken from the processed or compressed preview stream. Store raw values unchanged and record the bit depth and saturation level (1023 for 10-bit) in the metadata, so processing knows where clipping is.
- The exposure time and gain actually reported by the camera for each frame are recorded in the metadata.

#### Motion

- With a limit switch at the front only, the backend homes at the start of every session and enforces the far travel limit in software.
- Every measurement position is approached from the same direction (away from the camera) to remove backlash.
- After each move the backend waits a configurable settle time before capturing (suggested default 0.5 s) so vibration from the move has died down.
- At the end of a session the backend re-homes and records the position (mm) where the limit switch triggers; any value other than 0 means steps were lost. (the stepper is open-loop).

## Data Storage

Capture always saves raw frames; all corrections happen in a separate, repeatable processing step (see Processing). This means processing can be changed later without re-collecting data.

Each session is stored in its own folder under `camera_node/data/`, named by timestamp and x/y offset, e.g. `2026-11-22T14-03-10_x+5.0_y-2.5/`:

- `metadata.json` containing:
  - x, y offsets (mm)
  - the z sampling plan: planned z (mm), and the actual z after rounding (mm) for every position
  - exposure time and gain of every frame
  - all measurement options used
  - start/end timestamps, end-of-session homing check result
  - application version / git commit hash
- Raw frames as 16-bit TIFF (or .npy), one file per frame, named by position index, pass and frame number, e.g. `z012_p1_f03.tiff`.
- Background frames named by exposure, e.g. `bg_e012000us_f00.tiff`.

Each frame is written to disk as soon as it is taken and metadata is updated incrementally, so a crash or abort does not lose the frames already collected.

### Z Sampling Plans

For sessions at different (x, y) offsets to form a grid, they must use the same z positions. The user can save a z sampling plan (min z, max z, number of positions, spacing) under a name and select it for later sessions. Sessions record which plan they used, and the Model tab warns when the selected sessions used different plans.

## Documentation

A markdown file should be provided describing the broad architecture of the project with separate and more detailed sections below describing information about the chosen file structure and what each file/group of files does in a clear, brief and organized format. It should also document the coordinate system, the data format, and how to run the application on the Pi and on a PC with the fake hardware. Code should be well documented with the layout summarized at the top of each file as necessary.

## Measurement Options

### Number of Captures in Z

The user should be able to input the number of positions in the Z direction (Default is 64).

### Linear vs Inverse-Depth Z Spacing

The user should be able to select whether the z positions are evenly spaced in z (linear) or evenly spaced in 1/z (Default is 1/z). Even spacing in 1/z gives roughly even changes in the PSF's scale, since the PSF magnification goes as approximately 1 + d/z (d being the diffuser-to-sensor distance).

Positions are rounded to the Arduinos' step resolution in mm, and the actual positions are stored (see Data Storage). The application should warn if any spacing is smaller than a few motor steps. (See the `controller` folder for Arduino code to verify resolution. You may read but not modify it.)

### Frames per Position

The number of frames captured at each position in each pass (Default is 5). These are averaged during processing to reduce noise.

### Number of Passes

The number of times the full z sweep is repeated (Default is 1). Multiple passes allow the processing step to detect and correct for frame-to-frame shifts of the PSF (see Processing, step 3).

### Background Measurement

The user should be able to select whether background (light-off) frames are taken to subtract residual/background light and the sensor's black level from the measurements (Default is ON).

- A background frame contains a fixed black level (independent of exposure) plus ambient light and dark current (which grow with exposure time). Therefore, a single background frame cannot simply be scaled to other exposure times. Background frames are taken at every exposure time used in the session, with several frames averaged (Default is the same as Frames per Position).
- The backend enters `waiting_for_user` and prompts the user to turn off the light, at the end of the session. After the background frames are taken, prompt the user to turn the light back on.

### Exposure Control

Camera auto exposure is never used. The user chooses one of:

- **Fixed exposure**: every frame uses the exposure time and gain set on the Live tab.
- **Per-position exposure** (Default): at each z position the backend picks an exposure time that keeps the brightest pixels below saturation (gain stays fixed at the gain set on the Live tab). This keeps a good signal across the large change in brightness between the near and far ends. User can set an exposure target and maximum exposure time (defaulting to the brightest pixel at 75% of the saturation level, measured above the black level and the cameras' hardware limit respectively). If the exposure reaches the maximum and the signal is still below a threshold, mark the position as 'no PSF', warn and move on.

In both modes, before each position is saved the backend checks for saturated pixels and warns (fixed mode) or lowers/raises the exposure and retakes (per-position mode). Raw frames are always stored unnormalized; normalization happens in processing.

## Processing

Processing runs from the Model tab on saved sessions and never modifies raw files. Each step is optional and selectable, and the processed result is cached with a record of which steps and parameters were used. Steps, in order:

1. **Background subtraction:** subtract the averaged background frame taken at the matching exposure time. Results are kept as floating point; negative values are not clipped until the final output.
2. **Frame averaging:** average the frames taken at each position within a pass.
3. **Pass registration and averaging:** when there are multiple passes, register each pass's frame at a given z to a reference frame using subpixel phase correlation (not the centroid, which is biased by background, vignetting and cropping), shift it into alignment, and average (or median) the aligned frames. Frames with unusually low sharpness or correlation (e.g. motion blur) can be excluded. Areas with no data after shifting are marked invalid in a mask (not filled with zeros, since they are unknown rather than dark) and excluded from the average.
   - The measured shifts are saved and shown. Random shifts between passes indicate vibration (correctable this way). A shift that is the same in every pass is a real offset of the source (e.g. the rail is not straight or not parallel to the optical axis) and should not be "corrected" away — it should be recorded as a correction to the source's actual x/y position at that z. As a check: at x = y = 0 the PSF should not drift with z; a steady drift means the rail is tilted.
4. **Normalization:** either divide by exposure time and analog gain (keeps the real brightness change with distance) or scale each PSF to sum to 1 (common for reconstruction). The user selects which.

## PSF Generator Model

The generator must not estimate an in-between PSF by blending neighboring measured PSFs pixel by pixel; blending two shifted caustic patterns gives two faint overlapping copies rather than one pattern in between. Instead, it uses the structure of a diffuser PSF:

- **Lateral (x, y):** moving the source sideways mostly shifts the PSF on the sensor, by approximately Δx·d/z. The generator takes the nearest measured PSF and shifts it by the predicted offset. The shift per mm at each z is fitted from the measured data (so d does not need to be known exactly). Measurements at other (x, y) offsets capture changes that a pure shift misses (vignetting, aberrations), so the nearest measured offset is always used as the starting point. If only one offset is available, use the predicted shift with d = 2.3 mm.
- **Depth (z):** the generator uses the nearest measured z plane. Scaling between planes (magnification approximately 1 + d/z) can be added later as an option.
