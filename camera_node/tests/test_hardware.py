"""Real-hardware checks. Skipped by default; run on the Pi with the app stopped:

    pytest -m hardware                 # camera and stage
    pytest -m hardware -k camera       # camera only
"""

import pytest

from psf_capture import config

pytestmark = pytest.mark.hardware


@pytest.fixture(scope="module")
def camera():
    from psf_capture.camera import RealCamera
    cam = RealCamera(10000, 1.0)
    yield cam
    cam.close()


def test_camera_raw_frames_are_full_resolution_10bit(camera):
    camera.set_controls(20000, 2.0)
    frames = camera.grab(3)
    for f in frames:
        assert f.raw.shape == (800, 1280) and f.raw.dtype.name == "uint16"
        assert f.raw.max() <= config.SATURATION
        assert abs(f.exposure_us - 20000) <= 20 and f.gain == pytest.approx(2.0)
    assert camera.black_level == 64
    assert camera.exposure_limits[1] > 1_000_000          # read after configure()
    assert camera.preview_jpeg()[:2] == b"\xff\xd8"


def test_camera_exposure_change_is_applied_before_frames_are_returned(camera):
    camera.set_controls(3000, 1.0)
    assert all(abs(f.exposure_us - 3000) <= 15 for f in camera.grab(2))
    camera.set_controls(40000, 1.0)
    assert all(abs(f.exposure_us - 40000) <= 50 for f in camera.grab(2))


@pytest.fixture(scope="module")
def stage():
    from psf_capture.stage import Stage, open_serial
    st = Stage(open_serial(config.SERIAL_PORT))
    yield st
    st.close()


def test_stage_home_move_and_homing_check(stage):
    stage.home()
    assert stage.z == config.Z_HOME_MM
    assert stage.move_to_z(30.0) == 30.0
    assert stage.move_to_z(25.5) == 25.5                 # approached from below
    result = stage.check_home()
    assert not result["steps_lost"], result
