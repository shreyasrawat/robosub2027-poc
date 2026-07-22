# RoboSub Ethernet Web Dashboard

## Context

The Jetson Orin Nano connects directly to a dev Mac over Ethernet (Jetson `192.168.2.2`,
Mac `192.168.2.1`). The operator wants to plug in a cable, open
`http://192.168.2.2:3000`, and immediately see: live ZED camera with object-detection
overlays, real-time telemetry, a 3D vehicle model, and mission/system status — with **no
software installed on the Mac** beyond a browser. Today the vision + telemetry code exists
only as standalone Python (`test_scripts/object_detection.py`) and an in-process control
API (`rov/`); nothing is exposed over the network. This adds a monitoring-only web layer
on top of the existing code.

**Scope decisions (confirmed with user):** monitoring only (no control commands sent to
the vehicle); camera as JPEG-over-WebSocket; frontend React + Vite + Three.js; add the
currently-missing telemetry fields (thrusters, leak, velocity, host stats) now, degrading
gracefully to N/A when a sensor is absent.

## Key constraints from the codebase

- **ZED = one handle per device per process.** Only ONE process may `grab()`. The
  dashboard backend therefore *owns* the ZED via the existing pipeline in
  `test_scripts/object_detection.py` (`CaptureStage`, `InferenceStage`, `detect()`,
  `get_session()` → native TensorRT). The full `rov` motion stack and standalone
  `object_detection.py` **cannot run at the same time** as the dashboard — document this.
- **MAVLink is a separate resource** from the ZED, so the same backend process can also
  hold a read-only MAVLink link. Telemetry is already modeled in
  `rov/api/telemetry.py` (`Telemetry.snapshot()` → depth/heading/roll/pitch/yaw/voltage/
  armed/mode). Reuse `rov.api.vehicle.Vehicle` + `rov.api.telemetry.Telemetry`.
- **Velocity comes from MAVLink**, not ZED — `GLOBAL_POSITION_INT`/`LOCAL_POSITION_NED`
  carry `vx/vy/vz`. This avoids enabling ZED positional tracking on the vision handle.
- `object_detection.py` is importable without touching the GPU until `get_session()` is
  called (per its docstring), so importing it as a library is safe.

## Architecture

New first-party top-level directory `dashboard/`:

```
dashboard/
  backend/
    app.py            FastAPI REST app (:8000) — health, config, CORS, quality/fps controls
    ws_server.py      websockets server (:8001) — /camera (binary JPEG) + /telemetry (JSON)
    vision_source.py  owns the ZED: reuses object_detection CaptureStage/InferenceStage,
                      draws boxes+labels+conf, cv2.imencode JPEG at runtime quality
    telemetry_source.py  wraps rov Vehicle+Telemetry, extra MAVLink handlers, host stats
    hoststats.py      CPU/RAM (psutil) + GPU/temp (jtop if present, else tegrastats/sysfs)
    settings.py       env-var config: HOST, PORTS, CORS_ORIGINS, JPEG_QUALITY, TARGET_FPS
    run_backend.py    entry: starts REST + WS servers + producer threads
  frontend/
    (Vite React app) src/components/{CameraView,TelemetryPanel,Vehicle3D,MissionStatus}
    src/hooks/useWebSocket.js   auto-reconnect wrapper
    src/config.js               reads import.meta.env.VITE_* with sane defaults
    .env / .env.example         VITE_API_URL, VITE_CAMERA_WS, VITE_TELEMETRY_WS
  scripts/            start-backend.sh, start-frontend.sh, start-all.sh (restart-on-crash)
  docker/             Dockerfile.backend, Dockerfile.frontend, docker-compose.yml
  README.md
```

### Data flow

1. **Vision producer thread** (`vision_source.py`): reuses `object_detection` pipeline —
   `CaptureStage.grab()` → `InferenceStage.run()` (native TRT). Draws boxes/labels/conf
   onto the frame (reuse `CLASS_NAMES`, `COLOR_BOX`; a small draw fn, not the full nav-HUD
   `render()`). Converts BGRA→BGR, `cv2.imencode('.jpg', …, [IMWRITE_JPEG_QUALITY, q])` (or
   WebP), stores newest JPEG + metadata (grab→encode `pipeline_ms`, server send time,
   detection list) in a drop-oldest `LatestSlot`.
2. **Telemetry producer thread** (`telemetry_source.py`): reuses `Vehicle` + `Telemetry`.
   Extends telemetry with new MAVLink handlers — `SERVO_OUTPUT_RAW` (thruster outputs),
   `GLOBAL_POSITION_INT` vx/vy/vz (velocity), `STATUSTEXT`/leak sensor (leak flag) — added
   in `rov/api/telemetry.py` (first-party, natural home) and surfaced in an extended
   snapshot. Merges host stats from `hoststats.py`. Missing fields → `null` → frontend N/A.
3. **WS server** (`ws_server.py`, :8001, asyncio): `/camera` pushes newest JPEG binary +
   small JSON header at `TARGET_FPS`; `/telemetry` pushes merged snapshot ~10–20 Hz.
   Bridges the producer threads (shared latest-value + event) to async sends. ping/pong
   for RTT.
4. **REST** (`app.py`, :8000): `GET /api/health`, `GET /api/config`, `POST /api/stream`
   (set JPEG quality / target FPS at runtime → updates `settings`), CORS from env.
5. **Frontend** (:3000): `CameraView` decodes binary JPEG frames from `/camera` WS into an
   `<img>`/canvas, shows FPS + end-to-end latency, sliders for quality/fps (POST to REST),
   auto-reconnect. `TelemetryPanel` renders the `/telemetry` JSON. `Vehicle3D` (Three.js)
   subscribes to telemetry roll/pitch/yaw and animates on its own `requestAnimationFrame`
   loop — **decoupled from camera FPS**. `MissionStatus` shows mode/armed/mission state.

### Networking / config / security

- Backend binds `0.0.0.0`; all hosts/ports from env (`settings.py`): `DASH_HOST`,
  `REST_PORT=8000`, `WS_PORT=8001`, `FRONTEND_PORT=3000`, `CORS_ORIGINS`.
- Frontend **never hardcodes localhost** — `config.js` derives URLs from
  `import.meta.env.VITE_*`, defaulting to `window.location.hostname` so it works both in
  Vite dev and served from the Jetson.
- CORS default restricts to the Ethernet origin (`http://192.168.2.2:3000` +
  `192.168.2.1`); no public binding unless `CORS_ORIGINS` widened. Document that this is
  LAN-only by design.

### Latency measurement

Each camera frame header carries server `send_time` (ms) and exact server-side
`pipeline_ms` (grab→encode). Frontend estimates clock offset from an initial WS
echo/ping, then shows approximate end-to-end latency = `client_now - (send_time - offset)`
plus the exact `pipeline_ms`. Document that the end-to-end figure is offset-corrected and
approximate.

## Files to modify / create

- **Create** everything under `dashboard/` (above).
- **Modify** `rov/api/telemetry.py`: add `thrusters`, `leak`, `vx/vy/vz` fields + handlers
  (`SERVO_OUTPUT_RAW`, `GLOBAL_POSITION_INT` velocity, `STATUSTEXT`) and include them in
  `snapshot()`. Additive only — existing accessors unchanged.
- **Modify** `CLAUDE.md`: add `dashboard/` to the Layout table and a short section; note
  the ZED single-handle exclusivity (dashboard vs. `object_detection.py`/full `rov` stack).
- **Reuse (no change):** `object_detection.py` (`CaptureStage`, `InferenceStage`,
  `detect`, `get_session`, `CLASS_NAMES`, `COLOR_BOX`, `LatestSlot`, `close_session`),
  `rov.api.vehicle.Vehicle`.

## Dependencies to add (Jetson, pip)

`fastapi`, `uvicorn[standard]`, `websockets`, `psutil` (host stats), optional
`jetson-stats` (jtop) for GPU/temp. Frontend: Node/npm for Vite + React + `three`.
Docker/Compose optional. Reuses existing `onnxruntime-gpu`/TensorRT/`pyzed`/`opencv`/
`pymavlink` already required by the repo.

## Verification (end-to-end)

1. **Telemetry unit** — extend/run `rov` offline checks: confirm extended `snapshot()`
   returns new keys with `None` when messages absent (no hardware).
2. **Backend standalone (no camera)** — run `run_backend.py` with ZED absent; confirm REST
   `/api/health` is up on `0.0.0.0:8000`, `/telemetry` WS pushes host stats + null vehicle
   fields, camera WS reports "no camera" gracefully.
3. **With ZED + Pixhawk connected on the Jetson** — start backend; from the Mac browser
   open `http://192.168.2.2:3000`: verify live overlays (boxes/labels/conf), telemetry
   values updating, 3D model reacting to attitude, latency readout, and that unplugging /
   replugging Ethernet auto-reconnects both WS channels.
4. **Deployment scripts** — `scripts/start-all.sh` brings up both services and restarts a
   killed one; `docker compose up` builds and serves the same. Confirm quality/fps sliders
   change the stream live.

## Out of scope

- Control commands from the browser (arm/move) — monitoring only this pass.
- Running dashboard concurrently with the full `rov` motion stack (ZED handle conflict).
- HTTPS/auth — LAN-only trust model.

---

# Implementation notes — deviations from the plan above

Everything above is the plan as approved before implementation. The sections below record
where the delivered system differs, so the plan is not read as a description of the code.

## 1. Attitude source changed: MAVLink → ZED positional tracking

The plan said velocity *and* orientation would come from MAVLink, explicitly to avoid
enabling ZED positional tracking on the vision handle. In practice the autopilot's
`ATTITUDE` stream arrived as `None` on this vehicle, so the 3D view had nothing to render,
and what did arrive was stale.

Delivered instead: `vision_source.py` enables `enable_positional_tracking()` on the single
ZED handle it already owns and reads the fused pose once per grab. `TelemetrySource`
exposes `set_attitude_provider()`; when a ZED orientation is available it overrides
`roll`/`pitch`/`yaw` in the published snapshot and adds `quat`, `zed_x/y/z`, and
`zed_tracking_ok`, plus a top-level `attitude_source` field (`"zed"` / `"mavlink"`).

The ZED orientation is used as soon as a quaternion exists, **not** gated on
`POSITIONAL_TRACKING_STATE.OK` — the SDK reports `SEARCHING` while re-aligning gravity, but
the IMU-fused orientation is already usable and still fresher than MAVLink. Validity is
published for the UI to flag rather than used to discard data.

**Velocity still comes from MAVLink** (`GLOBAL_POSITION_INT`), as planned.

## 2. 3D orientation uses a quaternion + change-of-basis, not euler angles

The plan said `Vehicle3D` would subscribe to roll/pitch/yaw. Euler angles required guessing
a rotation order and three signs, and the axes came out wrong.

Delivered: the backend sends the ZED orientation quaternion, and the frontend rotates it
into the Three.js frame with a fixed change-of-basis — vehicle (X fwd, Y left, Z up) →
Three (X right, Y up, Z back), applied as `q_three = qB · q_zed · qB⁻¹`. The model is built
nose-along −Z so an identity pose renders level and forward-facing. Per-axis easing was
replaced with `slerp` (no gimbal artefacts). Euler remains as a fallback path.

## 3. Frontend must bind IPv4 explicitly

Not anticipated by the plan. `vite --host` binds `::` (IPv6 only), so the Mac's IPv4
requests to `192.168.2.2:3000` timed out while the IPv4-bound backend on `:8000` worked.
The `--host` flag was removed from the `dev`/`preview` npm scripts so `vite.config.js`'s
`host: '0.0.0.0'` wins. Verify with `ss -tlnp | grep :3000` — it must show `0.0.0.0`, not `*`.

## 4. Host stats must be polled off the telemetry thread

Not anticipated by the plan. `jtop.ok()` **blocks ~1 s per call** (it waits for the jtop
service's next sample). Calling it inside the 15 Hz telemetry loop stalled every snapshot
and froze the dashboard on a single stale sample — which presents to an operator as severe
lag, not as failure.

`HostStats` now runs its own poll thread with a cached result; `snapshot()` never blocks,
and `ok()`'s blocking becomes that thread's natural pacing. Two related defects were fixed
at the same time: a Tegra thermal-zone read raising `TypeError` (not caught by the original
`except (OSError, ValueError)`) killed the telemetry thread outright, and the telemetry loop
had no exception guard at all. Both are now handled — the loop logs and continues.

**Rule to preserve:** nothing called from the telemetry loop may block.

## 5. WebSocket write backpressure

Added beyond the plan: `websockets.serve(..., write_limit=32768)`. Without a bounded write
buffer, a client that drains slowly accumulates a queue of stale snapshots and the
dashboard appears laggy. With the limit, `await send()` applies backpressure and — because
each iteration re-reads the latest snapshot after the await — a slow client receives fewer
samples that are always current.

## 6. Firewall

Not a code change, but required on the Jetson: the host firewall must permit the dashboard
ports on the direct-Ethernet subnet, or connections time out (silently dropped, not
refused):

```bash
sudo ufw allow from 192.168.2.0/24 to any port 3000 proto tcp
sudo ufw allow from 192.168.2.0/24 to any port 8000 proto tcp
sudo ufw allow from 192.168.2.0/24 to any port 8001 proto tcp
```

The `from 192.168.2.0/24` scoping is deliberate — a bare `allow 3000` would open the port
on every interface including Wi-Fi, defeating the LAN-only design.

## Verification status

Steps 1, 2 and 4 of the plan's verification were run and passed. Step 3 was run on the
Jetson with the ZED and Pixhawk attached: telemetry staleness measured 5–68 ms (avg 46 ms)
at a steady 15 Hz, `attitude_source=zed` with live roll/pitch/yaw, camera 1280×720 at
~62–70 ms pipeline with detections, and all host stats populated.

**Not verified:** that the 3D model's rendered orientation matches the physical vehicle's
attitude. The axis mapping is correct by construction, but it needs a human eye on the real
sub to confirm.
