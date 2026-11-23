"""HTTP API with fake hardware: locking (409), session flow, model build and downloads."""

import io
import json

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

from psf_capture.server import create_app

from conftest import wait_for


@pytest.fixture
def client(backend):
    return TestClient(create_app(backend))


def test_frontend_and_status(client):
    assert "alpine" in client.get("/").text.lower()
    s = client.get("/api/status").json()
    assert s["state"] == "idle" and s["fake"] and not s["homed"]
    d = client.get("/api/defaults").json()
    assert d["options"]["n_positions"] == 64 and d["options"]["spacing"] == "inverse"
    assert d["options"]["frames_per_position"] == 5 and d["options"]["background"] is True
    assert d["options"]["exposure_mode"] == "per_position" and d["z_range"] == [10, 340]


def test_live_controls_and_stats(client):
    assert client.post("/api/stage/move", json={"z": 50}).status_code == 500   # not homed yet
    client.post("/api/stage/home")
    s = client.post("/api/stage/move", json={"z": 55.5}).json()
    assert s["z_mm"] == pytest.approx(55.5)
    r = client.post("/api/camera/controls", json={"exposure_us": 1234, "gain": 2.03}).json()
    assert r == {"exposure_us": 1234, "gain": 2.0}
    st = client.get("/api/live/stats").json()
    assert len(st["histogram"]) == 128 and "saturated_px" in st


def test_plans_preview_and_storage(client):
    prev = client.post("/api/plans/preview", json={"z_min": 10, "z_max": 340, "n_positions": 64,
                                                   "spacing": "inverse"}).json()
    assert len(prev["actual_z"]) == 64 and prev["actual_z"][0] == 10 and prev["actual_z"][-1] == 340
    gaps = np.diff(prev["actual_z"])
    assert gaps[0] < gaps[-1]                                       # denser close to the diffuser
    bad = client.post("/api/plans/preview", json={"z_min": 10, "z_max": 10.01, "n_positions": 20,
                                                  "spacing": "linear"}).json()
    assert any("motor steps" in w for w in bad["warnings"])
    client.post("/api/plans", json={"name": "grid", "z_min": 20, "z_max": 300, "n_positions": 8,
                                    "spacing": "linear"})
    assert client.get("/api/plans").json()["grid"]["n_positions"] == 8
    assert client.post("/api/plans/preview", json={"z_min": 5, "z_max": 300, "n_positions": 3}).status_code == 400


def test_session_locking_and_model_flow(client, backend):
    client.post("/api/plans", json={"name": "grid", "z_min": 30, "z_max": 90, "n_positions": 3,
                                    "spacing": "inverse"})
    common = {"plan_name": "grid", "z_min": 30, "z_max": 90, "n_positions": 3, "spacing": "inverse",
              "frames_per_position": 1, "settle_s": 0}
    assert client.post("/api/session/start", json={"z_min": 30}).status_code == 400   # x, y missing
    sid = client.post("/api/session/start", json={**common, "x_mm": 0, "y_mm": 0}).json()["session_id"]
    assert client.post("/api/stage/home").status_code == 409
    assert client.post("/api/camera/controls", json={"exposure_us": 100, "gain": 1}).status_code == 409
    assert client.post("/api/model/generate", json={"points": []}).status_code == 409
    wait_for(lambda: backend.state == "waiting_for_user")
    assert client.get("/api/status").json()["prompt"]["id"] == "light_off"
    client.post("/api/session/continue")
    wait_for(lambda: backend.state == "waiting_for_user" and backend.prompt["id"] == "light_on")
    client.post("/api/session/continue")
    wait_for(lambda: backend.state == "idle")

    backend.start_session({**common, "x_mm": 1.5, "y_mm": 0.5, "background": False})
    wait_for(lambda: backend.state == "idle")
    sessions = client.get("/api/sessions").json()
    assert len(sessions) == 2 and all(s["plan_name"] == "grid" for s in sessions)
    assert sessions[-1]["id"] == sid

    client.post("/api/model/build", json={"sessions": [s["id"] for s in sessions]})
    wait_for(lambda: not backend.job["running"], timeout=300)
    job = client.get("/api/status").json()["job"]
    assert job["error"] is None, job["error"]
    assert job["result"]["summary"]["fitted_d_mm"] == pytest.approx(2.3, rel=0.1)

    r = client.post("/api/model/generate", json={"points": [{"x": 0.5, "y": 0.2, "z": 50},
                                                            {"x": 9, "y": 0, "z": 50}]}).json()
    assert any("outside the measured x range" in w for w in r["warnings"])
    arr = np.load(io.BytesIO(client.get("/api/model/result.npy").content))
    assert arr.shape == (800, 1280) and arr.dtype == np.float32 and arr.min() >= 0
    png = Image.open(io.BytesIO(client.get("/api/model/result.png").content))
    assert png.mode.startswith("I") and np.array(png).max() == 65535
    tif = tifffile.imread(io.BytesIO(client.get("/api/model/result.tiff").content))
    assert tif.dtype == np.uint16 and tif.max() == 65535
    assert client.get("/api/model/preview.png").status_code == 200


def test_sse_pushes_snapshots(client, backend):
    with client.stream("GET", "/api/events") as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                assert json.loads(line[5:])["state"] == "idle"
                break
