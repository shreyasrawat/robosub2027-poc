# RoboSub 2027 Operator Dashboard

Monitoring dashboard hosted **on the Jetson**, viewed from the Mac over the
direct Ethernet cable. Plug in, open `http://192.168.2.2:3000`, and see the live
ZED feed with detection overlays, telemetry, a 3D attitude model, and mission
status. Nothing needs installing on the Mac beyond a browser.

**Read-only.** The dashboard never arms, never sets modes, never commands
thrusters. The MAVLink link is monitoring-only.

```
Mac (192.168.2.1)  ──Ethernet──  Jetson (192.168.2.2)
  browser :3000  ◄── HTTP ──  frontend   (Vite/React, :3000)
                 ◄── REST ──  backend    (FastAPI,    :8000)
                 ◄──  WS  ──  ws server  (            :8001)
                                 │         /camera  /telemetry  /ping
                                 ├─ vision_source     → owns the ZED: detection + pose
                                 └─ telemetry_source  → MAVLink (read-only) + host stats
```

---

## ⚠ Before you start: the ZED is exclusive

The ZED SDK allows **one grabber per device per process**. The dashboard takes
it. While the dashboard is running you **cannot** run:

- `test_scripts/object_detection.py`
- the full `rov` motion stack (it opens the ZED via `zed_pose`)

Starting either one alongside the dashboard fails to open the camera. Stop the
dashboard first — see [Stopping](#stopping-and-releasing-the-zed).

MAVLink is a *separate* resource, so the read-only telemetry link is fine in the
same process.

---

## One-time setup

Do these once per Jetson. Steps 3 and 4 are the two people forget, and both
produce a dashboard that looks completely unreachable.

### 1. Python dependencies

```bash
cd ~/robosub2027/robosub-2027-ws
pip3 install -r dashboard/backend/requirements.txt
```

This installs only the web layer (`fastapi`, `uvicorn`, `websockets`,
`pydantic`, `psutil`). It deliberately does **not** touch the board-specific
stack — `pyzed`, `onnxruntime-gpu`, TensorRT, `opencv-python`, `pymavlink` must
already be installed per the root `CLAUDE.md`.

Optional, for GPU load and component temperatures:

```bash
sudo pip3 install jetson-stats     # provides jtop; needs a reboot or service start
```

Without it, host stats fall back to `psutil` + Tegra sysfs (CPU/RAM/GPU/temp
still work, just less detail).

### 2. Frontend dependencies

```bash
cd dashboard/frontend && npm install && npm run build
```

`npm run build` is optional — the launch script builds automatically if `dist/`
is missing — but doing it now makes the first launch much faster.

### 3. Open the firewall (required)

The Jetson's firewall silently **drops** traffic to unopened ports, so the Mac
sees a connection *timeout*, not a refusal. Open the three dashboard ports,
scoped to the direct-Ethernet subnet:

```bash
sudo ufw allow from 192.168.2.0/24 to any port 3000 proto tcp
sudo ufw allow from 192.168.2.0/24 to any port 8000 proto tcp
sudo ufw allow from 192.168.2.0/24 to any port 8001 proto tcp
sudo ufw status | grep -E "3000|8000|8001"      # confirm
```

Keep the `from 192.168.2.0/24` scoping. A bare `sudo ufw allow 3000` opens the
port on **every** interface including Wi-Fi, which defeats the LAN-only design.

If `ufw status` says inactive, the rules live in raw iptables/nftables instead —
inspect with `sudo iptables -S INPUT`.

### 4. Configure the Mac's Ethernet interface

A direct cable has no DHCP server, so the Mac needs a static IP.
**System Settings → Network → (your USB/Ethernet adapter) → Configure IPv4:
Manually**

| Field | Value |
|---|---|
| IP Address | `192.168.2.1` |
| Subnet Mask | `255.255.255.0` |
| Router | *(leave blank)* |

Verify from the Mac: `ping 192.168.2.2` should reply.

---

## Launching

### Normal use — both services, auto-restart

```bash
cd ~/robosub2027/robosub-2027-ws
dashboard/scripts/start-all.sh
```

Starts backend and frontend under a supervisor that restarts either one if it
crashes. `Ctrl-C` stops both. Startup takes **~25–30 s** — the ZED opens, the
TensorRT engine loads, and positional tracking initialises.

To leave it running after you close the terminal:

```bash
setsid nohup dashboard/scripts/start-all.sh > /tmp/dashboard.log 2>&1 < /dev/null &
```

### Individually

```bash
dashboard/scripts/start-backend.sh          # REST :8000 + WS :8001
dashboard/scripts/start-frontend.sh         # build + serve :3000
dashboard/scripts/start-frontend.sh dev     # Vite dev server with hot reload
```

### Without hardware (bench testing)

```bash
DASH_ENABLE_VISION=0 DASH_ENABLE_TELEMETRY=0 dashboard/scripts/start-backend.sh
```

REST and both WS channels still come up; the camera channel reports its status
and telemetry streams host stats with `null` vehicle fields. Useful for frontend
work with no ZED or Pixhawk attached.

### Verbose logs

```bash
DASH_LOG_LEVEL=DEBUG dashboard/scripts/start-backend.sh
```

---

## Verifying it works

Run these **on the Jetson** first — they isolate a Jetson problem from a network
or browser problem.

```bash
# 1. Are all three ports listening, and on IPv4 (0.0.0.0)?
ss -tlnp | grep -E ":3000|:8000|:8001"
```

Every line must show `0.0.0.0`. A `*` on `:3000` means IPv6-only — see
[Troubleshooting](#troubleshooting).

```bash
# 2. Is the backend healthy, and did the camera and link come up?
curl -s http://192.168.2.2:8000/api/health
# {"ok":true,"camera_status":"running","link_status":"connected"}

# 3. Does the frontend serve?
curl -s -o /dev/null -w "%{http_code}\n" http://192.168.2.2:3000/    # 200
```

`camera_status` values: `running`, `starting`, `no camera`, `disabled`, or an
error string. `link_status`: `connected`, `stopped`, or `no link: <reason>`.

Then **from the Mac**:

```bash
nc -vz 192.168.2.2 3000        # succeeded
nc -vz 192.168.2.2 8000        # succeeded
nc -vz 192.168.2.2 8001        # succeeded
```

All three must succeed. `:8001` is easy to forget and the failure is confusing —
the page loads fine but the video and telemetry stay blank.

Finally, open **`http://192.168.2.2:3000`** in the browser.

Use `http://`, not `https://` — there is no TLS server, and Chrome will
sometimes auto-upgrade. Include the `:3000`; nothing listens on port 80, so a
bare `http://192.168.2.2` reads as unreachable.

### Checking telemetry is actually live

Frozen telemetry looks exactly like laggy telemetry. The snapshot carries a `t`
timestamp — if `t` does not advance, a producer is stuck, not slow:

```bash
python3 - <<'EOF'
import asyncio, json, time, websockets
async def main():
    async with websockets.connect("ws://192.168.2.2:8001/telemetry") as ws:
        for i in range(5):
            d = json.loads(await ws.recv())
            print(f"t={d['t']:.0f} staleness={time.time()*1000-d['t']:.0f}ms "
                  f"src={d['attitude_source']} cpu={d['host']['cpu']}")
            await asyncio.sleep(1)
asyncio.run(main())
EOF
```

Healthy: `t` advances ~67 ms per sample, staleness under ~100 ms,
`src=zed`.

---

## Stopping and releasing the ZED

`Ctrl-C` in the `start-all.sh` terminal stops everything. If it is running
detached, kill the **supervisors first** or they will respawn the backend:

```bash
# Supervisors first, then the backend that holds the camera
pkill -f start-all.sh
sleep 1
pkill -f run_backend.py
```

Confirm the camera is actually free before starting `object_detection.py` or the
`rov` stack:

```bash
fuser /dev/video0        # no output = released
ss -tlnp | grep -E ":3000|:8000|:8001"    # no output = all stopped
```

If `fuser` still prints a PID, that process holds the ZED — kill it by PID.

---

## Configuration

All backend settings are environment variables (`dashboard/backend/settings.py`):

| Variable | Default | Meaning |
|---|---|---|
| `DASH_HOST` | `0.0.0.0` | Bind address |
| `DASH_REST_PORT` | `8000` | REST API port |
| `DASH_WS_PORT` | `8001` | WebSocket port |
| `DASH_FRONTEND_PORT` | `3000` | Frontend port (also read by `vite.config.js`) |
| `DASH_CORS_ORIGINS` | LAN hosts | Comma-separated allowed browser origins |
| `DASH_JPEG_QUALITY` | `70` | Initial encode quality (adjustable at runtime) |
| `DASH_TARGET_FPS` | `20` | Initial stream FPS (adjustable at runtime) |
| `DASH_IMAGE_FORMAT` | `jpeg` | `jpeg` or `webp` |
| `DASH_TELEMETRY_HZ` | `15` | Telemetry push rate |
| `DASH_ENABLE_VISION` | `1` | `0` = telemetry-only, no ZED |
| `DASH_ENABLE_TELEMETRY` | `1` | `0` = host-stats-only, no MAVLink |
| `DASH_LOG_LEVEL` | `INFO` | `DEBUG` for verbose |
| `MAV_DEVICE` / `MAV_BAUD` | `/dev/ttyACM0` / `115200` | MAVLink link (shared with `rov`) |

Frontend (`dashboard/frontend/.env`, all optional — see `.env.example`):
`VITE_API_URL`, `VITE_WS_URL`. When unset, every URL derives from the browser's
own hostname, so the same build works in dev and on the Jetson. **localhost is
never hardcoded.**

Quality and FPS are also adjustable live from the sliders in the UI.

---

## API reference

### REST — `http://192.168.2.2:8000`

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | `{ok, camera_status, link_status}` |
| `GET /api/config` | Public settings |
| `POST /api/stream` | `{jpeg_quality?, target_fps?}` — live stream tuning |

### WebSocket — `ws://192.168.2.2:8001`

- **`/camera`** — binary: `[4-byte BE header length][JSON header][image bytes]`.
  Header carries `send_time`, `pipeline_ms`, `format`, `width`, `height`, and
  `detections[]` (`class_id`, `label`, `conf`, `box`). Metadata travels with the
  pixels so the client never has to correlate two streams.
- **`/telemetry`** — JSON: `{t, link_status, mission_state, attitude_source,
  vehicle{...}, host{...}}`. Absent sensors are `null` → the UI shows N/A.
- **`/ping`** — send anything, receive `{server_time}`; used to estimate the
  clock offset behind the camera latency readout.

Attitude note: `vehicle.roll/pitch/yaw` come from the **ZED's fused pose** when
available (`attitude_source: "zed"`), with `quat`, `zed_x/y/z`, and
`zed_tracking_ok` alongside. Velocity comes from MAVLink.

---

## Troubleshooting

Symptom-first, because in this system almost every failure presents as something
other than its cause.

| Symptom | Cause | Fix |
|---|---|---|
| Chrome: "unreachable", instantly | URL missing `:3000`, or forced `https://` | Use `http://192.168.2.2:3000` exactly; try Incognito to clear an HSTS upgrade |
| `curl`/`nc` **times out** on a port | Firewall dropping it (drops are silent; refusals are instant) | Add the `ufw` rule for that port — see [setup step 3](#3-open-the-firewall-required) |
| `:8000` works but `:3000` times out | Only 8000 was opened, **or** Vite bound IPv6-only | Open 3000; check `ss -tlnp \| grep :3000` shows `0.0.0.0`, not `*` |
| Page loads, but video and telemetry blank | Port `8001` not open | Open 8001 — it carries both WS channels |
| Telemetry values frozen but page responsive | A producer thread is stuck, not slow | Check `t` advances (see [above](#checking-telemetry-is-actually-live)); nothing in the telemetry loop may block |
| `camera_status: "no camera"` | ZED held by another process, or unplugged | `fuser /dev/video0`, kill the holder; confirm the ZED is connected |
| `camera_status` shows a detector error | TensorRT engine missing or wrong board | Rebuild `ffc_rs_26.engine` per root `CLAUDE.md` — engines are board-specific |
| `link_status: "no link: ..."` | Pixhawk not on `MAV_DEVICE` | Check `ls /dev/ttyACM*`, set `MAV_DEVICE` |
| 3D model frozen but telemetry updates | ZED tracking has no pose yet | Check `attitude_source`; `zed_tracking_ok` false during gravity re-alignment is normal at startup |
| Ping works, TCP does not | Firewall (ICMP is allowed, TCP is not) | Per-port `ufw` rules |

### Why "unreachable" is usually not the network

The Jetson-side check is `curl http://192.168.2.2:3000/` **on the Jetson**. If
that returns `200`, the service and its binding are fine, and the problem lives
in the firewall, the Mac's interface config, or the URL. Working outward in that
order is much faster than guessing.

---

## Docker (optional)

```bash
cd dashboard/docker && docker compose up --build
```

The backend container needs the NVIDIA runtime plus ZED (USB) and Pixhawk
(serial) passthrough; see `docker-compose.yml`. It uses `network_mode: host` so
the ports bind the Jetson's Ethernet IP directly. Native launch via
`start-all.sh` is the tested path — Docker is provided as an alternative.

---

## Security

LAN-only by design:

- CORS is restricted to the Ethernet origins.
- Firewall rules are scoped to `192.168.2.0/24`.
- There is **no authentication and no HTTPS.**

Do not expose these ports to the public internet or to an untrusted network. If
you need to widen access, change `DASH_CORS_ORIGINS` deliberately and understand
that anyone who can reach the port can view the feed and telemetry.

---

## Further reading

- Root `CLAUDE.md` — where the dashboard sits in the workspace, plus the
  non-obvious constraints to preserve when editing.
- `plans/dashboard-planning.md` — the original design and where the
  implementation deviated from it, with reasons.
