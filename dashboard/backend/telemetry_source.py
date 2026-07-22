"""Vehicle telemetry + host stats producer for the dashboard.

Holds a read-only MAVLink link via the existing ``rov`` stack
(:class:`rov.api.vehicle.Vehicle` + :class:`rov.api.telemetry.Telemetry`) and
merges the autopilot snapshot with Jetson host metrics (:mod:`hoststats`) into a
single dict published over the telemetry WebSocket.

Orientation is the exception: when a ZED attitude provider is registered (see
:meth:`TelemetrySource.set_attitude_provider`), roll/pitch/yaw come from the
ZED's fused visual-inertial pose instead of MAVLink ATTITUDE. The ZED value is
per-frame fresh and axis-consistent with the camera frame, and on this vehicle
the MAVLink attitude stream is both slower and frequently absent.

MAVLink is a distinct resource from the ZED, so this coexists with
:mod:`vision_source` in one process. The link is *read-only*: this never arms,
sets modes, or sends control — the dashboard is monitoring-only.

The MAVLink device/baud come from the existing ``rov`` config (``MAV_DEVICE`` /
``MAV_BAUD`` env vars), so the dashboard shares the vehicle's link settings.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from settings import SETTINGS  # noqa: E402
from hoststats import HostStats  # noqa: E402

logger = logging.getLogger("dashboard.telemetry")


class TelemetrySource:
    """Background poller that snapshots vehicle + host state.

    ``snapshot()`` is cheap and lock-free from the caller's view (the underlying
    ``Telemetry.snapshot`` and ``HostStats.snapshot`` each guard their own
    state), so the WebSocket layer can call it at its own cadence.
    """

    def __init__(self) -> None:
        self._host = HostStats()
        self._vehicle = None
        self._telemetry = None
        self._link_status = "stopped"
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._latest: dict = {}
        self._lock = threading.Lock()
        self._mission_state = "IDLE"  # settable by mission code via set_mission_state
        self._attitude_provider = None

    @property
    def link_status(self) -> str:
        return self._link_status

    def set_mission_state(self, state: str) -> None:
        with self._lock:
            self._mission_state = str(state)

    def set_attitude_provider(self, provider) -> None:
        """Register a callable returning the latest ZED attitude dict (or None).

        When it yields an orientation, roll/pitch/yaw in the published vehicle
        snapshot come from the ZED's fused VIO (fresh, per-frame,
        axis-consistent) instead of the MAVLink ATTITUDE stream, and the
        orientation quaternion is added for the 3D view.
        """
        self._attitude_provider = provider

    def start(self) -> None:
        self._host.start()
        if SETTINGS.enable_telemetry:
            self._connect_vehicle()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._host.stop()
        if self._telemetry is not None:
            try:
                self._telemetry.stop()
            except Exception:
                pass
        if self._vehicle is not None:
            try:
                self._vehicle.disconnect()
            except Exception:
                pass

    def _connect_vehicle(self) -> None:
        try:
            from rov.api.config import CONFIG
            from rov.api.vehicle import Vehicle
            from rov.api.telemetry import Telemetry
        except Exception as exc:
            self._link_status = f"rov import failed: {exc}"
            logger.error("could not import rov stack: %s", exc)
            return
        self._vehicle = Vehicle(CONFIG)
        self._telemetry = Telemetry(self._vehicle, CONFIG)
        try:
            self._telemetry.start()        # subscribe before traffic starts
            self._vehicle.connect()
            self._link_status = "connected"
            logger.info("telemetry link connected")
        except Exception as exc:
            self._link_status = f"no link: {exc}"
            logger.warning("MAVLink connect failed (%s); telemetry will be host-only", exc)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        period = 1.0 / max(1, SETTINGS.telemetry_hz)
        while not self._stop.is_set():
            # Never let one bad sample kill the loop. A dead telemetry thread
            # silently freezes the dashboard at its last value, which reads to
            # an operator as "telemetry is laggy" rather than "telemetry is
            # gone" — the worst possible failure mode for a monitoring UI.
            try:
                snap = self._build_snapshot()
                with self._lock:
                    self._latest = snap
            except Exception:
                logger.exception("telemetry snapshot failed; continuing")
            self._stop.wait(period)

    def _build_snapshot(self) -> dict:
        vehicle = {}
        if self._telemetry is not None:
            try:
                vehicle = self._telemetry.snapshot()
            except Exception as exc:
                logger.debug("telemetry snapshot failed: %s", exc)
        host = self._host.snapshot()

        # Prefer ZED-fused attitude for orientation (fresher + correct axes).
        att_source = "mavlink"
        if self._attitude_provider is not None:
            try:
                att = self._attitude_provider()
            except Exception:
                att = None
            # Use the ZED orientation as soon as there is one. Requiring
            # POSITIONAL_TRACKING_STATE.OK is too strict: the SDK reports
            # SEARCHING while it re-aligns gravity, yet the IMU-fused
            # orientation is already good and is still far fresher than
            # MAVLink ATTITUDE. Validity is published, not used to discard.
            if att and att.get("quat"):
                vehicle["roll"] = att["roll"]
                vehicle["pitch"] = att["pitch"]
                vehicle["yaw"] = att["yaw"]
                vehicle["quat"] = att["quat"]
                vehicle["zed_x"] = att["x"]
                vehicle["zed_y"] = att["y"]
                vehicle["zed_z"] = att["z"]
                vehicle["zed_tracking_ok"] = att["valid"]
                att_source = "zed"

        with self._lock:
            mission = self._mission_state
        return {
            "t": time.time() * 1000.0,
            "link_status": self._link_status,
            "mission_state": mission,
            "attitude_source": att_source,
            "vehicle": vehicle,
            "host": host,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._latest) if self._latest else self._build_snapshot()
