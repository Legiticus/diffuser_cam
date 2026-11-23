"""Processing steps on synthetic data with known answers."""

import numpy as np
import pytest

from psf_capture import processing
from psf_capture.processing import phase_correlate, process_session, shift_image

from synthetic import camera, make_session


def fourier_shift(img, dy, dx):
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (fy * dy + fx * dx))).real


@pytest.fixture(scope="module")
def pattern():
    return camera().render(0, 0, 40)


@pytest.mark.parametrize("dy,dx", [(0.3, -0.7), (4.25, 11.6), (-25.5, 2.1)])
def test_phase_correlation_recovers_known_subpixel_shift(pattern, dy, dx):
    got = phase_correlate(pattern, fourier_shift(pattern, dy, dx))
    assert got[0] == pytest.approx(dy, abs=0.05)
    assert got[1] == pytest.approx(dx, abs=0.05)
    assert got[2] > 0.8


def test_phase_correlation_is_not_fooled_by_background_gradient(pattern):
    ramp = np.linspace(0, 3 * pattern.max(), pattern.shape[1])[None, :]   # vignetting-like
    got = phase_correlate(pattern, fourier_shift(pattern, 2.4, -1.2) + ramp)
    assert got[:2] == pytest.approx((2.4, -1.2), abs=0.1)


def test_shift_image_marks_uncovered_pixels_invalid(pattern):
    out = shift_image(pattern, 2.5, -10)
    assert np.isnan(out[:2]).all() and np.isnan(out[:, -10:]).all()
    assert not np.isnan(out[3:, :-11]).any()
    assert out[103, 500] == pytest.approx(0.5 * (pattern[100, 510] + pattern[101, 510]), rel=1e-4)


def test_background_subtraction_and_frame_average(tmp_path):
    d, truth = make_session(tmp_path, ambient=7.0, noise=2.0, frames=4)
    out = process_session(d, {"normalize": "none"})
    psf, info = processing.load(out)
    assert info["steps"]["background"] == "frames"
    err = np.abs(psf[1] - truth[(1, 1)]).mean()
    assert err < 1.0                                 # noise 2 averaged over 4 frames, no offset
    assert np.mean(psf[1] - truth[(1, 1)]) == pytest.approx(0, abs=0.1)
    # black level only: the 7 counts of ambient light remain
    psf2, _ = processing.load(process_session(d, {"background": "black_level", "normalize": "none"}))
    assert np.mean(psf2[1] - truth[(1, 1)]) == pytest.approx(7.0, abs=0.2)


def test_normalisation(tmp_path):
    d, truth = make_session(tmp_path, exposure=1500)
    by_exposure, _ = processing.load(process_session(d, {"normalize": "exposure"}))
    assert by_exposure[0].max() == pytest.approx(truth[(0, 1)].max() / 1500, rel=0.02)
    summed, _ = processing.load(process_session(d, {"normalize": "sum"}))
    for plane in summed:
        assert np.nansum(plane) == pytest.approx(1.0, rel=1e-4)


def test_pass_registration_recovers_jitter_and_masks_edges(tmp_path):
    jitter = {1: (0.0, 0.0), 2: (1.6, -0.8), 3: (-0.4, 2.0)}
    d, truth = make_session(tmp_path, passes=3, pass_shift=lambda p, i: jitter[p])
    psf, info = processing.load(process_session(d, {"normalize": "none"}))
    mean = np.mean(list(jitter.values()), axis=0)
    for entry in info["pass_shifts"]:
        for q in entry["passes"]:
            want = np.array(jitter[q["pass"]]) - mean
            assert (q["dy"], q["dx"]) == pytest.approx(tuple(want), abs=0.1)
            assert not q["excluded"]
    # result sits at the mean position, sharp (not three smeared copies)
    expected = camera().render(0, 0, 30.0, tuple(mean)) * 1000
    ok = ~np.isnan(psf[0])
    assert np.corrcoef(psf[0][ok], expected[ok])[0, 1] > 0.98
    assert np.isnan(psf[0]).any()                       # edges with no data are invalid, not 0


def test_blurred_pass_is_excluded(tmp_path):
    d, _ = make_session(tmp_path, passes=3, blur_pass=2, z=(30.0,))
    _, info = processing.load(process_session(d))
    excluded = {q["pass"]: q["excluded"] for q in info["pass_shifts"][0]["passes"]}
    assert excluded == {1: False, 2: True, 3: False}


def test_cache_is_reused_and_records_steps(tmp_path):
    d, _ = make_session(tmp_path, z=(30.0, 60.0))
    a = process_session(d, {"combine": "median"})
    b = process_session(d, {"combine": "median"})
    c = process_session(d, {"combine": "mean"})
    assert a == b and a != c
    assert processing.load(a)[1]["steps"]["combine"] == "median"


def test_drift_with_z_gives_source_offset_corrections(tmp_path):
    # even in 1/z; the PSF stays inside the sensor (a clipped PSF biases its centroid,
    # which is used as the scaling centre)
    z = tuple(float(v) for v in np.round(1 / np.linspace(1 / 30, 1 / 150, 10), 3))
    # straight rail at the entered position: no correction needed
    good, _ = make_session(tmp_path, z=z, frames=1, seed=5, x=2.0, y=-1.0)
    _, info = processing.load(process_session(good))
    assert info["drift_fit"]["apparent_offset_mm"]["x_mm"] == pytest.approx(2.0, abs=0.03)
    assert info["drift_fit"]["apparent_offset_mm"]["y_mm"] == pytest.approx(-1.0, abs=0.03)
    for d in info["drift"]:
        assert abs(d["source_dx_mm"]) < 0.05 and abs(d["source_dy_mm"]) < 0.05
    # entered (0, 0) but the source really sits at x = 1.2 mm: the PSF drifts with z
    # and every plane reports a +1.2 mm correction
    off, _ = make_session(tmp_path, z=z, frames=1, seed=6, x=0.0, source_x=lambda zz: 1.2)
    _, info = processing.load(process_session(off))
    assert abs(info["drift_fit"]["drift_px"][1]) == pytest.approx(1.2 * 2.3 / 0.003 * (1 / 30 - 1 / 150), rel=0.03)
    assert info["drift_fit"]["offset_error_mm"]["x_mm"] == pytest.approx(1.2, abs=0.03)
    for d in info["drift"]:
        assert d["source_dx_mm"] == pytest.approx(1.2, abs=0.05)
    # a kink in the rail shows up only at the z where it happens
    kink, _ = make_session(tmp_path, z=z, frames=1, seed=7, x=0.0,
                           source_x=lambda zz: 0.5 if zz == z[5] else 0.0)
    _, info = processing.load(process_session(kink))
    corr = {d["z_mm"]: d["source_dx_mm"] for d in info["drift"]}
    assert corr[z[5]] - np.median(list(corr.values())) > 0.3
