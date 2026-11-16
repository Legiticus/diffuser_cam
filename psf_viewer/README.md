# PSF viewer

Interactive check of a `psf_capture` run: pick a point in front of the camera and
see the PSF the sensor should record for it.

```bash
cd psf_viewer
python3 app.py                                   # latest run in camera_node/data/
python3 app.py ../camera_node/data/psf_YYYYMMDD_HHMMSS --t-mm 6.2
```

Open `http://<pi>:8003/`, click a point on the object-plane map, move the depth
slider, and the predicted sensor image updates. `download .npy` saves the
prediction (float32, counts per microsecond).

## How the prediction is made

DiffuserCam model, with `t` = diffuser-to-sensor distance and `d` = source-to-sensor distance:

- **Depth:** the nearest captured PSF (in 1/d) is magnified by `M(d)/M(d_i)`, with
  `M(d) = d / (d - t)`. Brightness falls as `1/(d - t)^2`.
- **Lateral position:** moving the point sideways by `(x, y)` mm moves the pattern on
  the sensor by `-(t / (d - t)) * (x, y)`, converted to pixels (3 µm pitch).
- PSFs are dark-subtracted (using the run's dark frames) and divided by their exposure.
- Areas shifted in from outside the captured sensor area are unknown and shown
  tinted blue, rather than made up.

`t` is estimated from the run by measuring how much each PSF is scaled relative to
the nearest one. If you know `t` from the build, pass `--t-mm`. It's faster, and
the estimate is unreliable when the PSFs are poor (the page shows a warning).

`x` and `y` follow the sensor's column and row directions, measured from the axis
the light source travels along.

## Files

| File | |
|---|---|
| `app.py` | Flask server and API (`/api/predict.png|json|npy`, `/api/fov.png|json`, `/api/info`, `/api/captured/<i>.png`) |
| `psf_model.py` | loading a run, estimating `t`, predicting PSFs, field-of-view map |
| `templates/index.html` | the page |
| `synthetic.py` | makes a synthetic run with a known `t` (`python3 synthetic.py /tmp/demo`) |
| `test_psf_model.py` | checks the model against the synthetic run (`python3 test_psf_model.py`) |
