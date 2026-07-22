"""WebSocket server for camera frames and telemetry (default :8001).

Two paths on one port:

* ``/camera``    — binary frames. Each message is a 4-byte big-endian header
  length, then a JSON header (``send_time``, ``pipeline_ms``, ``format``,
  ``width``, ``height``, ``detections``), then the raw JPEG/WebP bytes. Pushed
  at the runtime ``target_fps``; a slow client simply receives the newest frame
  (drop-oldest), never a backlog.
* ``/telemetry`` — JSON snapshots at ``telemetry_hz``.
* ``/ping``      — echoes ``{"server_time": ms}`` so the client can estimate the
  clock offset used for the approximate end-to-end camera latency readout.

The frame framing keeps metadata and pixels in one binary message so the client
never has to correlate two streams. The producers run on their own threads; the
async handlers just read the latest value and send it, so a stalled socket can
never stall acquisition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time

import websockets

from settings import SETTINGS

logger = logging.getLogger("dashboard.ws")


def _path_of(websocket, fallback) -> str:
    # websockets >=11 exposes websocket.request.path; older passes path as arg.
    req = getattr(websocket, "request", None)
    if req is not None and getattr(req, "path", None):
        return req.path
    return fallback or "/"


class WSServer:
    def __init__(self, vision, telemetry) -> None:
        self._vision = vision
        self._telemetry = telemetry

    async def handler(self, websocket, path=None):
        route = _path_of(websocket, path).rstrip("/") or "/"
        peer = getattr(websocket, "remote_address", None)
        logger.info("ws client %s connected to %s", peer, route)
        try:
            if route.endswith("/camera"):
                await self._camera(websocket)
            elif route.endswith("/telemetry"):
                await self._telemetry_stream(websocket)
            elif route.endswith("/ping"):
                await self._ping(websocket)
            else:
                await websocket.close(code=1008, reason=f"unknown path {route}")
        except websockets.ConnectionClosed:
            pass
        except Exception:
            logger.exception("ws handler error on %s", route)
        finally:
            logger.info("ws client %s disconnected from %s", peer, route)

    # ------------------------------------------------------------------
    async def _camera(self, websocket) -> None:
        if self._vision is None:
            await websocket.send(_frame(b"", {"status": "disabled"}))
            return
        last_seq = -1
        while True:
            fps = max(1, SETTINGS.target_fps)
            jpeg, meta, seq = await asyncio.to_thread(
                self._vision.latest.wait_newer, last_seq, 1.0)
            if seq == last_seq or jpeg is None:
                # No new frame within the wait window — send a keepalive status
                # so the client can show why the feed is dark.
                await websocket.send(_frame(b"", {"status": self._vision.status}))
                continue
            last_seq = seq
            await websocket.send(_frame(jpeg, meta))
            await asyncio.sleep(1.0 / fps)

    async def _telemetry_stream(self, websocket) -> None:
        while True:
            snap = self._telemetry.snapshot() if self._telemetry else {"link_status": "disabled"}
            await websocket.send(json.dumps(snap))
            await asyncio.sleep(1.0 / max(1, SETTINGS.telemetry_hz))

    async def _ping(self, websocket) -> None:
        async for _ in websocket:
            await websocket.send(json.dumps({"server_time": time.time() * 1000.0}))

    async def serve(self) -> None:
        # write_limit bounds how much unsent data may queue per client before
        # `await send()` applies backpressure. Left at the default, a client
        # that drains slowly accumulates a backlog of stale snapshots and the
        # dashboard appears laggy. With a small limit the sender blocks instead,
        # and because each loop re-reads the *latest* snapshot after the await,
        # a slow client receives fewer samples that are always current.
        async with websockets.serve(
            self.handler, SETTINGS.host, SETTINGS.ws_port,
            max_size=None, ping_interval=20, ping_timeout=20,
            write_limit=32768,
        ):
            logger.info("WebSocket server on ws://%s:%d (/camera /telemetry /ping)",
                        SETTINGS.host, SETTINGS.ws_port)
            await asyncio.Future()  # run forever


def _frame(payload: bytes, meta: dict) -> bytes:
    header = json.dumps(meta).encode("utf-8")
    return struct.pack(">I", len(header)) + header + payload
