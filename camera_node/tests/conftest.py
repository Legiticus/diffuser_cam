"""Shared fixtures: a Backend wired to the fake stage and fake camera in a temp data dir."""

import time

import pytest

from psf_capture.camera import FakeCamera, FakeWorld
from psf_capture.core import Backend
from psf_capture.stage import FakeArduino, Stage


@pytest.fixture
def backend(tmp_path):
    fw = FakeArduino()
    world = FakeWorld(fw)
    b = Backend(FakeCamera(world), Stage(fw), tmp_path / "data", world=world)
    yield b
    b._abort.set()
    b._continue.set()


def wait_for(predicate, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    raise TimeoutError("condition not reached")


def run_session(backend, params, answer_prompts=True, timeout=120.0):
    """Start a session and keep pressing Continue until it is idle again."""
    sid = backend.start_session(params)
    end = time.monotonic() + timeout
    while backend.state != "idle":
        if backend.state == "error":
            raise RuntimeError(backend.error)
        if backend.state == "waiting_for_user" and answer_prompts:
            backend.continue_prompt()
        assert time.monotonic() < end, "session did not finish"
        time.sleep(0.02)
    return sid
