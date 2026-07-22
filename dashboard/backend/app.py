"""FastAPI REST surface for the dashboard (default :8000).

Small, monitoring-only HTTP API:

* ``GET  /api/health``  — liveness + producer status (camera / link).
* ``GET  /api/config``  — the public settings the frontend needs.
* ``POST /api/stream``  — set JPEG/WebP quality and target FPS at runtime.

CORS is restricted to the LAN origins from :data:`settings.SETTINGS.cors_origins`
so a browser on the public internet cannot call the API even if the port were
somehow reachable. The heavy lifting (camera + telemetry) is on the WebSocket
server; this app only carries control-plane requests.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from settings import SETTINGS


class StreamSettings(BaseModel):
    jpeg_quality: int | None = None
    target_fps: int | None = None


def create_app(vision, telemetry) -> FastAPI:
    """Build the REST app wired to the running producers."""
    app = FastAPI(title="RoboSub Dashboard API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SETTINGS.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "camera_status": vision.status if vision else "disabled",
            "link_status": telemetry.link_status if telemetry else "disabled",
        }

    @app.get("/api/config")
    def config() -> dict:
        return SETTINGS.public()

    @app.post("/api/stream")
    def stream(body: StreamSettings) -> dict:
        return SETTINGS.update_stream(
            jpeg_quality=body.jpeg_quality, target_fps=body.target_fps)

    return app
