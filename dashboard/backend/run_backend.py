"""Dashboard backend entry point.

Brings up, in one process:

* the vision producer (owns the ZED, runs detection + pose) — :mod:`vision_source`,
* the telemetry producer (MAVLink + host stats)             — :mod:`telemetry_source`,
* the REST app on ``DASH_REST_PORT`` (default 8000)         — :mod:`app`,
* the WebSocket server on ``DASH_WS_PORT`` (default 8001)   — :mod:`ws_server`.

REST (uvicorn) and the WebSocket server share one asyncio event loop; the two
producers run on their own threads. Everything binds ``DASH_HOST`` (default
``0.0.0.0``) so it is reachable over the direct Ethernet link. ``Ctrl-C`` tears
producers down cleanly (thrusters are never commanded — this is read-only).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Ensure this directory is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

from settings import SETTINGS
from app import create_app
from ws_server import WSServer
from vision_source import VisionSource
from telemetry_source import TelemetrySource

logging.basicConfig(
    level=os.environ.get("DASH_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("dashboard")


async def _main() -> None:
    vision = VisionSource()
    telemetry = TelemetrySource()
    # Attitude for the 3D view comes from the ZED's fused pose (read on the
    # vision thread, which owns the only ZED handle), not the MAVLink ATTITUDE
    # stream. Wire the provider before either producer starts.
    telemetry.set_attitude_provider(vision.get_attitude)
    vision.start()
    telemetry.start()

    app = create_app(vision, telemetry)
    ws = WSServer(vision, telemetry)

    uvicorn_config = uvicorn.Config(
        app, host=SETTINGS.host, port=SETTINGS.rest_port,
        log_level=os.environ.get("DASH_LOG_LEVEL", "info").lower(),
        access_log=False,
    )
    server = uvicorn.Server(uvicorn_config)

    logger.info("REST on http://%s:%d  |  WS on ws://%s:%d  |  frontend expected on :%d",
                SETTINGS.host, SETTINGS.rest_port, SETTINGS.host, SETTINGS.ws_port,
                SETTINGS.frontend_port)

    try:
        await asyncio.gather(server.serve(), ws.serve())
    finally:
        vision.stop()
        telemetry.stop()


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
