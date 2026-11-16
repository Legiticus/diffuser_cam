# PSF viewer

Shows the most recent `psf_capture` run and predicts the PSF the sensor would record
for a point anywhere in front of the camera.

```bash
cd psf_viewer
python3 app.py                       # follow the newest run in camera_node/data/
python3 app.py --t-mm 2.6            # known diffuser-to-sensor distance (skips estimating t)
python3 app.py --run ../camera_node/data/psf_YYYYMMDD_HHMMSS   # stay on one run
```

Open `http://<pi>:8003/`. Click a point on the object-plane map (or type x, y), set the
depth d, and the predicted sensor image updates. `download .npy` saves the prediction
(float32, counts per microsecond).

## Always the latest data

The viewer checks `camera_node/data/` every 2 s (`--poll`). When a newer run appears, or
the current run's manifest changes (a new PSF during a sweep, or dark frames at the
end), it reloads in the background and the page updates by itself. A banner says when it
switches to a new run, and the header says whether the run is still capturing.

New data shows up right away. Estimating t takes about 20 s on the Pi, so until it
finishes the previous estimate is used, and the new one is swapped in when ready.
Estimates are cached in `~/.cache/diffuser_cam/psf_viewer/geometry.json`, so restarting
the viewer is instant. When a run is too short to estimate t (its first PSF), the most
recent reliable estimate from an earlier run is used, and the page says which run it
came from.

## How the prediction is made

DiffuserCam model, with `t` = diffuser-to-sensor distance and `d` = source-to-sensor distance:

- **Depth:** the two captures either side of d (in 1/d) are each magnified by
  `M(d)/M(d_i)`, with `M(d) = d / (d - t)`, about the *scale centre* (where the source
  axis meets the sensor), then blended linearly in 1/d. Outside the captured range the
  nearest end capture is used, and the page marks the depth as extrapolated.
- **Brightness:** a smooth fit of total captured light vs depth, not an ideal 1/z² law.
  The real runs fall off noticeably slower than 1/z².
- **Lateral position:** moving the point sideways by `(x, y)` mm moves the pattern on the
  sensor by `-(t / (d - t)) * (x, y)`, converted to pixels (3 µm pitch).
- PSFs are dark-subtracted using the run's dark frames, or a flat background estimate
  before the darks exist, and divided by their exposure.
- Areas shifted in from outside the captured sensor area are unknown. They're tinted
  blue rather than made up.

`t` and the scale centre are estimated from the run: up to 6 PSFs are compared with the
nearest one to measure how much each is scaled and offset, then `t` is fitted to the
scale-vs-depth model and the centre to the offsets. On `psf_20261116_111557` this gives
t ≈ 2.56 mm and a centre near pixel (1030, 492). If the PSFs are poor the estimate is
flagged on the page. Then set `t` in the page or pass `--t-mm`.

`x` and `y` follow the sensor's column and row directions, measured from the axis
the light source travels along. The lateral model hasn't been checked against real
off-axis captures yet: the capture rig only moves the source in depth.

## Accuracy on real data

`test_psf_model.py` hides captures of the newest real run one at a time and predicts
them from the rest. On `psf_20261116_111557` the predictions match the held-out
captures with a normalised correlation of 0.95–0.99. Brightness is usually within 5%,
but up to 14% off below ~45 mm, where exposures are only ~20–40 µs.

## Files

| File | |
|---|---|
| `app.py` | Flask server and API (`/api/state`, `/api/info`, `/api/predict.png\|json\|npy`, `/api/fov.png\|json`, `/api/captured/<i>.png`) |
| `runs.py` | follows the newest run, reloads on changes, caches t estimates |
| `psf_model.py` | loading captures, estimating t and the scale centre, predicting PSFs, field-of-view map |
| `templates/index.html` | the page |
| `synthetic.py` | synthetic runs with a known `t` (`python3 synthetic.py /tmp/demo/psf_20260101_000000`) |
| `test_psf_model.py` | tests (`python3 test_psf_model.py`); about 2.5 min on the Pi |
