"""Stage logic against the emulated firmware: conversions, rounding, limits, approach direction."""

import pytest

from psf_capture import coords
from psf_capture.config import STEPS_PER_MM, Z_HOME_MM, Z_MAX_MM
from psf_capture.stage import FakeArduino, Stage, StageError


class RecordingArduino(FakeArduino):
    """Remembers every MOVE argument so the approach direction can be checked."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.moves = []

    def _handle(self, line):
        if line.startswith("MOVE"):
            self.moves.append(float(line.split()[1]))
        super()._handle(line)


def make_stage(**kw):
    fw = RecordingArduino(**kw)
    stage = Stage(fw)
    stage.home()
    return stage, fw


def test_z_slider_conversion():
    assert coords.z_to_steps(Z_HOME_MM) == 0
    assert coords.steps_to_z(800) == pytest.approx(Z_HOME_MM + 1)
    assert coords.steps_to_slider_mm(800) == -1.0          # Arduino sign is opposite to z
    assert coords.round_z(12.3456) == pytest.approx(12.345)  # nearest 1/800 mm


def test_home_and_move_lands_on_exact_steps():
    stage, fw = make_stage()
    assert stage.z == Z_HOME_MM
    for z in [12.34625, 100.0, 37.00125, 339.99875, 10.00125]:
        reached = stage.move_to_z(z)
        assert reached == pytest.approx(coords.round_z(z))
        assert fw.z == pytest.approx(reached)                 # truncation bias works


def test_positions_are_always_approached_moving_away_from_camera():
    stage, fw = make_stage()
    stage.move_to_z(200)
    fw.moves.clear()
    stage.move_to_z(150)
    # first an overshoot towards the camera (+), then the final move away (-)
    assert fw.moves[0] > 0 and fw.moves[-1] < 0
    assert fw.moves[0] * STEPS_PER_MM == pytest.approx(50 * STEPS_PER_MM + 800 + 0.5)


def test_software_limits_and_unknown_position():
    fw = FakeArduino()
    stage = Stage(fw)
    with pytest.raises(StageError):
        stage.move_to_z(50)              # not homed
    stage.home()
    with pytest.raises(StageError):
        stage.move_to_z(Z_MAX_MM + 0.1)
    with pytest.raises(StageError):
        stage.move_to_z(Z_HOME_MM - 0.1)


def test_switch_probe_detects_lost_steps():
    stage, fw = make_stage()
    baseline = stage.probe_switch()
    assert baseline == pytest.approx(fw.HOME_BACKOFF / STEPS_PER_MM)   # switch hysteresis
    assert stage.z == Z_HOME_MM
    stage.move_to_z(300)
    assert stage.probe_switch() == pytest.approx(baseline)             # nothing lost

    fw.lose_steps = 100                    # every move travels 100 steps short
    for z in [100, 200, 300]:
        stage.move_to_z(z)
    assert abs(stage.probe_switch() - baseline) > 0.05
