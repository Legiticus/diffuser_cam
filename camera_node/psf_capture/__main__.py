"""Entry point:  python -m psf_capture [--fake | --fake-camera | --fake-stage] [options]

Builds the camera and stage (real or fake), the Backend, and serves the app with
uvicorn on port 8003 (API + frontend from one server, so no CORS setup).
"""

import argparse
import logging

import uvicorn

from . import config
from .camera import FakeCamera, FakeWorld
from .core import Backend
from .server import create_app
from .stage import FakeArduino, Stage, open_serial


def main():
    ap = argparse.ArgumentParser(description="PSF capture web app")
    ap.add_argument("--fake", action="store_true", help="fake camera and fake stage (no hardware)")
    ap.add_argument("--fake-camera", action="store_true")
    ap.add_argument("--fake-stage", action="store_true")
    ap.add_argument("--fake-speed", type=float, default=1.0,
                    help="time scale of the fakes (1 = real time, 0 = instant)")
    ap.add_argument("--serial", default=config.SERIAL_PORT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=config.HTTP_PORT)
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    fw = None
    if args.fake or args.fake_stage:
        fw = FakeArduino(speed_scale=args.fake_speed)
        stage = Stage(fw)
    else:
        stage = Stage(open_serial(args.serial))

    world = None
    exposure, gain = config.DEFAULTS["exposure_us"], config.DEFAULTS["gain"]
    if args.fake or args.fake_camera:
        world = FakeWorld(fw)
        camera = FakeCamera(world, exposure, gain, time_scale=args.fake_speed)
    else:
        from .camera import RealCamera
        camera = RealCamera(exposure, gain)

    backend = Backend(camera, stage, args.data_dir, world=world)
    try:
        uvicorn.run(create_app(backend), host=args.host, port=args.port, log_level="warning")
    finally:
        camera.close()
        stage.close()


if __name__ == "__main__":
    main()
