"""PSF generator: lateral shift fit, nearest plane, summing and range warnings."""

import numpy as np
import pytest

from psf_capture import processing
from psf_capture.model import PsfModel

from synthetic import camera, make_session

Z = (40.0, 80.0)


@pytest.fixture(scope="module")
def two_offsets(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    a, _ = make_session(tmp, x=0.0, y=0.0, z=Z, frames=1, seed=1)
    b, _ = make_session(tmp, x=2.0, y=1.0, z=Z, frames=1, seed=2)
    return [processing.process_session(a), processing.process_session(b)]


def test_fitted_shift_per_mm_matches_geometry(two_offsets):
    model = PsfModel(two_offsets)
    for z in Z:
        assert model.shift_per_mm(z) == pytest.approx(processing.shift_per_mm(z), rel=0.03)
    assert model.summary()["fitted_d_mm"] == pytest.approx(2.3, rel=0.03)


def test_generated_psf_is_shifted_nearest_measurement(two_offsets):
    model = PsfModel(two_offsets)
    img, warnings, details = model.generate([{"x": 0.6, "y": 0.2, "z": 41.0}])
    assert warnings == []
    assert details[0]["z_plane_mm"] == 40.0
    expected = camera().render(0.6, 0.2, 40.0) * 1   # exposure-normalised units
    assert np.corrcoef(img.ravel(), expected.ravel())[0, 1] > 0.95


def test_points_are_summed_with_intensity(two_offsets):
    model = PsfModel(two_offsets)
    one, _, _ = model.generate([{"x": 0, "y": 0, "z": 40}])
    two, _, _ = model.generate([{"x": 0, "y": 0, "z": 40, "intensity": 2.0},
                                {"x": 0, "y": 0, "z": 40}])
    assert two.sum() == pytest.approx(3 * one.sum(), rel=1e-5)


def test_out_of_range_points_warn(two_offsets):
    model = PsfModel(two_offsets)
    _, warnings, _ = model.generate([{"x": 5, "y": -3, "z": 300}])
    text = " ".join(warnings)
    assert "x=5.0" in text and "y=-3.0" in text and "z=300.0" in text


def test_single_session_uses_predicted_shift_and_plan_mismatch_warns(tmp_path, two_offsets):
    single = PsfModel(two_offsets[:1])
    assert single.k_fit == {}
    assert single.shift_per_mm(50) == pytest.approx(processing.shift_per_mm(50, 2.3))
    other, _ = make_session(tmp_path, z=(40.0, 90.0), frames=1, plan_name="other", seed=9)
    mixed = PsfModel([two_offsets[0], processing.process_session(other)])
    assert any("different z sampling plans" in w for w in mixed.warnings)
