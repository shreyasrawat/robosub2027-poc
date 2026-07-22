"""Environment-driven configuration for the dashboard backend.

Every network binding and tunable is read from an environment variable with a
LAN-safe default. Nothing is hardcoded to a specific host in the frontend; the
backend binds ``0.0.0.0`` so it is reachable over the direct Ethernet link
(Jetson ``192.168.2.2``), while CORS restricts *browser* origins to the LAN.

The stream-quality knobs (``jpeg_quality``, ``target_fps``) are mutable at
runtime through the REST ``POST /api/stream`` endpoint; the rest are read once
at startup.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import List


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _csv(name: str, default: str) -> List[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    """Resolved backend configuration.

    Fields map 1:1 to ``DASH_*`` environment variables. ``jpeg_quality`` and
    ``target_fps`` are guarded by a lock because the REST layer mutates them
    while the vision producer thread reads them.
    """

    host: str = field(default_factory=lambda: os.environ.get("DASH_HOST", "0.0.0.0"))
    rest_port: int = field(default_factory=lambda: _int("DASH_REST_PORT", 8000))
    ws_port: int = field(default_factory=lambda: _int("DASH_WS_PORT", 8001))
    frontend_port: int = field(default_factory=lambda: _int("DASH_FRONTEND_PORT", 3000))

    #: Browser origins allowed by CORS. Defaults to the direct-Ethernet hosts.
    cors_origins: List[str] = field(default_factory=lambda: _csv(
        "DASH_CORS_ORIGINS",
        "http://192.168.2.2:3000,http://192.168.2.1:3000,http://localhost:3000",
    ))

    #: Encode format for camera frames: "jpeg" or "webp".
    image_format: str = field(default_factory=lambda: os.environ.get("DASH_IMAGE_FORMAT", "jpeg"))
    _jpeg_quality: int = field(default_factory=lambda: _int("DASH_JPEG_QUALITY", 70))
    _target_fps: int = field(default_factory=lambda: _int("DASH_TARGET_FPS", 20))
    telemetry_hz: int = field(default_factory=lambda: _int("DASH_TELEMETRY_HZ", 15))

    #: Whether to open the ZED and run detection. Set 0 to run telemetry-only
    #: (used for standalone backend testing without a camera).
    enable_vision: bool = field(default_factory=lambda: os.environ.get("DASH_ENABLE_VISION", "1") != "0")
    #: Whether to connect the MAVLink link. Set 0 for host-stats-only testing.
    enable_telemetry: bool = field(default_factory=lambda: os.environ.get("DASH_ENABLE_TELEMETRY", "1") != "0")

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    # -- runtime-mutable stream knobs (thread-safe) --------------------------
    @property
    def jpeg_quality(self) -> int:
        with self._lock:
            return self._jpeg_quality

    @property
    def target_fps(self) -> int:
        with self._lock:
            return self._target_fps

    def update_stream(self, jpeg_quality: int = None, target_fps: int = None) -> dict:
        """Clamp and apply runtime stream settings; returns the new values."""
        with self._lock:
            if jpeg_quality is not None:
                self._jpeg_quality = max(1, min(100, int(jpeg_quality)))
            if target_fps is not None:
                self._target_fps = max(1, min(60, int(target_fps)))
            return {"jpeg_quality": self._jpeg_quality, "target_fps": self._target_fps}

    def public(self) -> dict:
        """Config surface safe to expose over REST (no secrets)."""
        return {
            "rest_port": self.rest_port,
            "ws_port": self.ws_port,
            "frontend_port": self.frontend_port,
            "image_format": self.image_format,
            "jpeg_quality": self.jpeg_quality,
            "target_fps": self.target_fps,
            "telemetry_hz": self.telemetry_hz,
            "enable_vision": self.enable_vision,
            "enable_telemetry": self.enable_telemetry,
        }


#: Process-wide singleton.
SETTINGS = Settings()
