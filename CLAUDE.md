# RoboSub 2027 Workspace

ROS2 (colcon) workspace for the RoboSub 2027 autonomous underwater vehicle.

## Layout

| Path | What |
|------|------|
| `rov/` | First-party. Autonomous control framework: MAVLink link, ZED localization, closed-loop motion, missions. See `rov/README.md`. |
| `dashboard/` | First-party. Web operator dashboard hosted on the Jetson: live ZED feed + detection overlays, telemetry, 3D attitude, mission status. FastAPI + WebSockets backend, React/Vite/Three.js frontend. See `dashboard/README.md`. |
| `src/zed-ros2-wrapper/` | Vendored Stereolabs ZED ROS2 wrapper (C++). Third-party; own git repo. |
| `test_scripts/` | First-party code. Standalone Python (no ROS runtime). |
| `test_scripts/runtime_info.py` | Inference-backend selection + startup GPU diagnostics. |
| `test_scripts/trt_backend.py` | Native TensorRT runtime for a prebuilt `.engine` (the fast path). |
| `plans/` | Plan / design documents for future work. |
| `build/` `install/` `log/` | colcon artifacts. Do not edit. |

Only `rov/`, `dashboard/`, `test_scripts/`, `plans/` and this file are first-party. Everything under `src/` is vendored.

## `rov/` — control framework

Layered API for autonomous operation. Mission code imports `rov` and never
touches PyMAVLink or the ZED SDK; the Pixhawk keeps all stabilization, motor
mixing and failsafes, and the package only sends `MANUAL_CONTROL` demands.

```
rov/api/config.py       every tunable (PID gains, tolerances, limits, timeouts)
rov/api/controllers.py  PID math (anti-windup, derivative-on-measurement)
rov/api/vehicle.py      the only pymavlink importer: link, arm, modes, MANUAL_CONTROL
rov/api/zed_pose.py     the only ZED SDK importer: fused VIO pose, velocity, tracking health
rov/api/telemetry.py    depth, attitude, battery, heartbeat (subscribes to vehicle)
rov/api/movement.py     closed-loop primitives: move_*, turn, goto, hold_*
rov/api/mission.py      behaviours: submerge, find_gate, drive_through, surface
rov/vision/             detections -> normalized control targets
rov/ros/                optional RViz bridge (publish-only), + rov.rviz config
rov/rov.py              the ROV facade
rov/main.py             CLI entry point
rov/tests/console.py    interactive movement REPL (safety-gated `sub` object)
```

```bash
python3 -m rov.main --status                    # telemetry only, never arms
python3 -m rov.main --mission square --side 0.5
python3 -m rov.main --status --rviz             # + publish to ROS2; then: rviz2 -d rov/ros/rov.rviz
python3 -m rov.tests.test_api                   # offline math checks, no hardware
python3 -m rov.tests.console                    # live REPL: sub.move_forward(0.5), sub.turn(90)
```

RViz shows pose/TF, IMU, heading, the live MANUAL_CONTROL demand and the
active goal on `/rov/*`. Cyan = measured, orange = commanded, green = goal.
The bridge never commands motion and `rclpy` is imported lazily, so the stack
still runs with no ROS installed.

Full design rationale in `rov/README.md`. Two properties to preserve when
editing: **every primitive is closed-loop** (no fixed-duration or sleep-based
motion anywhere), and **one MAVLink reader / one camera grabber** — pymavlink
connections are unsafe to read concurrently and the ZED SDK allows a single
handle per device per process.

Adds `pymavlink` to the dependency set below. Gains are untuned starting
values and the mount offsets are placeholder zeros.

## `dashboard/` — web operator dashboard

Monitoring-only web UI hosted on the Jetson, viewed from the Mac over the direct
Ethernet link (Jetson `192.168.2.2`, Mac `192.168.2.1`). Plug in the cable, open
`http://192.168.2.2:3000`.

```
dashboard/backend/    FastAPI REST (:8000) + websockets server (:8001)
  vision_source.py    owns the ZED; reuses object_detection's CaptureStage/
                      InferenceStage/detect + CLASS_NAMES; encodes JPEG/WebP
  telemetry_source.py wraps rov Vehicle+Telemetry (read-only) + host stats
  hoststats.py        CPU/RAM (psutil) + GPU/temp (jtop, else Tegra sysfs)
  settings.py         DASH_* env config; runtime-mutable quality/fps
dashboard/frontend/   React + Vite + Three.js (CameraView, TelemetryPanel,
                      Vehicle3D, MissionStatus); URLs from window.location
dashboard/scripts/    start-backend|frontend|all.sh (restart-on-crash)
dashboard/docker/     optional Dockerfile.{backend,frontend} + compose
```

- **ZED single-handle exclusivity:** the dashboard owns the one ZED grabber, so
  do **not** run `object_detection.py` or the full `rov` motion stack (which
  opens the ZED via `zed_pose`) concurrently. MAVLink is a separate resource, so
  the read-only telemetry link coexists fine in the same process.
- **Attitude comes from the ZED, velocity from MAVLink.** `vision_source` enables
  positional tracking on the one ZED handle and reads the fused pose every grab;
  `TelemetrySource.set_attitude_provider()` then overrides `roll/pitch/yaw` in the
  snapshot and adds `quat`, `zed_x/y/z`, `zed_tracking_ok`, `attitude_source`.
  Reason: this vehicle's MAVLink `ATTITUDE` arrives as `None`. The ZED orientation
  is used as soon as a quaternion exists — **not** gated on
  `POSITIONAL_TRACKING_STATE.OK`, since the SDK reports `SEARCHING` while
  re-aligning gravity but the orientation is already good. Velocity still comes
  from MAVLink `GLOBAL_POSITION_INT`.
- `rov/api/telemetry.py` gained `thrusters` (`SERVO_OUTPUT_RAW`), `velocity`, and
  `leak` (`STATUSTEXT`) fields + `snapshot()` keys; additive, existing accessors
  unchanged.
- Camera frames stream as one binary WS message (`[4B len][JSON header][image]`)
  so overlays/latency metadata travel with the pixels. Missing telemetry sensors
  serialize as `null` → the UI shows N/A.
- **3D orientation is a quaternion, not euler.** The frontend rotates the ZED
  quaternion into the Three.js frame with a fixed change-of-basis — vehicle
  (X fwd, Y left, Z up) → Three (X right, Y up, Z back), `q3 = qB·q·qB⁻¹` — and
  the model is built nose-along `-Z`. Euler angles required guessing a rotation
  order plus three signs and rendered the axes wrong; do not "simplify" it back.
- LAN-only by design: CORS restricted to the Ethernet origins; no auth/HTTPS.
  Adds `fastapi`, `uvicorn`, `websockets`, `psutil` (+ optional `jetson-stats`)
  to the Python deps, and Node/npm for the frontend build.

### Gotchas that cost real debugging time

- **Nothing called from the telemetry loop may block.** `jtop.ok()` waits for the
  jtop service's next sample (~1 s measured). Calling it in the 15 Hz loop froze
  every snapshot, which presents to an operator as *severe lag*, not as failure.
  `HostStats` therefore polls on its own thread and `snapshot()` only reads a
  cache. The loop also wraps each snapshot in `try/except` — a dead producer
  thread silently replays its last value forever.
- **Tegra sysfs reads need a broad `except`.** Some `thermal_zone*/temp` nodes
  return `None` mid-read and raise `TypeError`, which `except (OSError, ValueError)`
  does not catch. That killed the telemetry thread outright.
- **Vite must bind IPv4.** `vite --host` binds `::` (IPv6 only), so IPv4 clients
  time out on `:3000` while the IPv4-bound backend on `:8000` works. The `--host`
  flag is deliberately absent from the npm scripts so `vite.config.js`'s
  `host: '0.0.0.0'` wins. Check with `ss -tlnp | grep :3000` → must be `0.0.0.0`,
  never `*`.
- **Host firewall must allow 3000/8000/8001** on the Ethernet subnet or the Mac
  times out (dropped, not refused):
  `sudo ufw allow from 192.168.2.0/24 to any port 3000 proto tcp` (repeat for
  8000/8001). Keep the subnet scoping — a bare `allow 3000` opens every interface.
- WS `write_limit=32768` bounds per-client buffering, so a slow client gets fewer
  but always-current samples instead of a growing queue of stale ones.

Full detail in `dashboard/README.md`; design + deviations in
`plans/dashboard-planning.md`.

## `test_scripts/object_detection.py`

Standalone ZED stereo detector + 3D target tracker + navigation-path overlay.

**Per-frame pipeline:** grab → retrieve LEFT image + XYZ point cloud → ONNX detect →
sample target's 3D point → transform to vehicle frame → track active class →
straight-line path → render boxes + nav arrow + path line + HUD.

Runs as three stages — **capture | inference | render** — in separate threads with
one-deep drop-oldest handoff (`LatestSlot`), so a slow stage drops frames instead of
stalling the others. `SERIAL=1` runs the original single-threaded loop; both paths call
the same stage functions. OpenCV HighGUI stays on the main thread (required).

### Model — `ffc_rs_26.onnx`
YOLOv8-seg, **NMS baked in**. Input `images [1,3,416,416]` (RGB, 0..1, CHW).
Two outputs; only `output0 [1,300,38]` is used:

| cols | meaning |
|------|---------|
| 0-3 | bbox **xyxy** corners, 416-px space |
| 4 | confidence |
| 5 | class id |
| 6-37 | 32 mask coefficients (unused) |

Rows padded to 300; unused rows have confidence 0. `output1 [1,32,104,104]` = mask
prototypes (unused — detection only, no mask rendering yet).

Classes: `0 blood, 1 buoy, 2 compass, 3 circle, 4 fire, 5 hammer_and_wrench, 6 slalom, 7 sos`.

### Inference backend (GPU)

`runtime_info.create_session()` picks the backend and prints a startup banner (available
providers, selected backend, GPU/board, JetPack, CUDA/cuDNN/TensorRT versions, and the
measured **fraction of graph nodes that actually executed on the GPU**). Order of
preference:

1. **Native TensorRT** (`trt_backend.py`) against a prebuilt `ffc_rs_26.engine` — fastest,
   smallest, ~0.5 s startup. Used automatically when the engine file is present.
2. ONNX Runtime + TensorRT provider — only if `USE_TENSORRT=1`; builds its own engine.
3. ONNX Runtime + CUDA provider.
4. CPU — rejected unless `REQUIRE_GPU=0`.

**Why a separate native backend:** ORT's TensorRT provider can only load engines *it*
built, keyed by an internal subgraph hash
(`TensorrtExecutionProvider_TRTKernel_graph_<hash>_0_0_fp16_sm87.engine`). A standalone
engine from `trtexec` or the TensorRT API is invisible to it, so asking ORT for TensorRT
would trigger an ~8 min on-device rebuild. `trt_backend.TensorRTSession` deserializes the
engine directly in ~0.5 s and drops the ORT graph, session and CUDA arena from the
process entirely.

`TensorRTSession` mimics the slice of `onnxruntime.InferenceSession` the detector uses
(`run`, `get_inputs`, `get_providers`), so `detect()` is backend-agnostic. It allocates
device buffers, page-locked host staging buffers and one CUDA stream **once**; a
steady-state frame does one H2D copy, one `execute_async_v3`, one D2H copy per output.
Pinned host memory matters — pageable memory forces the driver through a bounce buffer,
roughly halving copy bandwidth. `object_detection.close_session()` releases it all.

**Engines are not portable.** A `.engine` is tied to the exact GPU architecture (sm87 for
Orin) and TensorRT version that built it. `*.engine` is gitignored; rebuild per board. If
deserialization fails the banner says so and the code falls back to ONNX Runtime rather
than dying.

**Requires `onnxruntime-gpu`, not `onnxruntime`.** The PyPI `onnxruntime` wheel is a
CPU-only build with no CUDA/TensorRT provider compiled in — asking for
`CUDAExecutionProvider` there only emits a `UserWarning` and silently runs on the CPU
(~133 ms/frame). Correct install for JetPack 6.2 / CUDA 12.6 / Python 3.10:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

`REQUIRE_GPU=1` (default) makes a CPU fallback a hard, explained failure rather than a
silent slowdown. Set `REQUIRE_GPU=0` to allow CPU.

| env var | default | meaning |
|---|---|---|
| `REQUIRE_GPU` | `1` | abort with a fix hint if the session lands on CPU |
| `USE_NATIVE_TRT` | `1` | use a prebuilt `.engine` directly; `0` forces ONNX Runtime |
| `TRT_ENGINE_PATH` | `test_scripts/ffc_rs_26.engine` | prebuilt engine to load |
| `USE_TENSORRT` | `0` | opt into ORT's TensorRT EP (see caveat below) |
| `TRT_FP16` | `1` | fp16 inference when TRT is used |
| `TRT_CACHE_DIR` | `test_scripts/.trt_cache` | cached engines (gitignored) |
| `GPU_MEM_LIMIT_BYTES` | 1 GiB | CUDA EP arena cap |
| `TRT_WORKSPACE_BYTES` | 512 MiB | TRT build workspace cap |
| `SERIAL` | `0` | `1` = single-threaded loop |
| `ZED_DEPTH_MODE` | `NEURAL` | e.g. `NEURAL_LIGHT` to cut depth GPU/VRAM cost |
| `PERF_LOG_EVERY` | `60` | frames between `[perf]` FPS/latency lines |

Measured inference latency, 416×416, this board:

| backend | latency | startup | process RSS | notes |
|---|---|---|---|---|
| CPU (old default) | **133 ms** | ~1 s | — | pegs all 6 cores |
| ORT CUDA EP | **18.5 ms** | ~2 s | — | no engine needed |
| ORT TensorRT fp16 | **10.1 ms** | **~8 min** first build | ~2.0 GB | rebuilds unless ORT's own cache hits |
| **Native TRT engine** | **6.7 ms** | **0.54 s** | **434 MB** | current default; 4.3 MB device memory |

**ORT's TensorRT EP is opt-in for a reason:** its first build takes ~8 minutes, drives the
8 GB board into swap, and can destabilise it when the ZED SDK and other tools are
resident. The native backend sidesteps this entirely — prefer it.

Startup banner on a healthy run:

```
==============================================================
  Inference execution environment
==============================================================
  backend            : native TensorRT engine (ffc_rs_26.engine)
  available providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
  requested (in order): NativeTensorRT
  SELECTED PROVIDER  : NativeTensorRT
  running on GPU     : YES
  GPU / board        : Orin (nvgpu)
  JetPack (L4T)      : R36 (release), REVISION: 4.7, ...
  CUDA / cuDNN       : 12.6 / 9.3.0
  TensorRT           : 10.3.0
  engine device mem  : 4.3 MB
  kernel placement   : 100% of graph nodes on GPU
==============================================================
```

Without an engine file the banner instead shows `SELECTED PROVIDER : CUDAExecutionProvider`
and a `[runtime] no prebuilt engine at ...; falling back to ONNX Runtime` line.

`kernel placement` is measured, not asserted: the warmup run is profiled and the ONNX
Runtime trace is parsed for each node's actual provider. Anything below 100% means part
of the graph fell back to the CPU.

### Measured live performance

CUDA EP, HD720@60, `DEPTH_MODE.NEURAL`, threaded pipeline. `[perf]` lines print every
`PERF_LOG_EVERY` frames (`capture`/`infer`/`render` = mean stage time, `latency` = grab →
on-screen):

```
[pipeline] mode=threaded depth=NEURAL
[perf]   7.1 FPS   capture  37.0ms infer  49.6ms render  85.8ms latency 200.9ms   <- warmup
[perf]  27.8 FPS   capture  36.4ms infer  33.0ms render   2.3ms latency  79.4ms
[perf]  27.4 FPS   capture  35.8ms infer  32.8ms render   2.2ms latency  72.2ms
```

**~7 FPS → ~27–28 FPS**, end-to-end latency ~75 ms. The first line is warmup (cuDNN plan
selection + first ZED depth frames); steady state arrives within ~60 frames.

Those numbers predate the native TensorRT backend, which cuts inference to 6.7 ms
(detect stage, pre+infer+post: 13.6 ms → 73 FPS standalone). Since capture already caps
the loop at ~28 FPS, the gain shows up as lower latency and freed GPU/CPU headroom for
the ZED depth network, not more frames.

**The bottleneck is now capture, not inference.** `grab()` with `NEURAL` depth plus the
`retrieve_measure(XYZRGBA)` device→host copy costs ~36 ms, which caps the pipeline at
~28 FPS no matter how fast the model gets. `infer` reads ~33 ms live versus 18.5 ms in
isolation because the ZED depth network and the detector share one GPU. Next levers, in
order of payoff:

1. `ZED_DEPTH_MODE=NEURAL_LIGHT` — cuts the dominant capture cost and frees GPU for inference.
2. `USE_TENSORRT=1` — ~10 ms inference, but only helps once capture is cheaper.
3. Retrieve the point cloud only on frames with an active-class detection (the serial
   path already does this via `wants_depth()`; the threaded capture stage does not,
   because it runs before detections exist).

### Per-frame optimizations

- **Preprocessing** (`Preprocessor`): resizes the 4-channel frame *first*, so the colour
  conversion runs on 416×416 instead of 1280×720, and folds alpha-drop + BGR→RGB into one
  `cvtColor`. All buffers (`_small`, `_rgb`, `_tensor`) are preallocated and reused —
  steady-state frames allocate nothing. 3.23 ms → **1.90 ms**.
- **Postprocessing**: the confidence threshold is applied as a mask over all 300 rows at
  once; only surviving rows (usually 0–5) become Python objects. The old per-row loop paid
  interpreter cost for ~295 zero-padding rows. ~1–2 ms → **0.012 ms**.
- **Path reprojection** (`project_many`): all `PATH_SAMPLES` points transform and project
  in two matrix ops instead of a Python loop; `_PATH_TS` is computed once at import.
- **Capture** (`CaptureStage`): two sets of `sl.Mat` alternate, so the consumer can read
  frame N while the camera fills N+1 without a copy or a torn read. `Mat.free()` on close.
- **Session** (`get_session`): built lazily and once, thread-safe. Importing the module
  for its geometry helpers (as `test_geometry.py` does) neither loads the model nor
  touches the GPU. `intra_op_num_threads=2` — ORT's default of one thread per core spent
  more time synchronising than computing and starved the ZED depth pipeline.
- **Handoff** (`LatestSlot`): one deep, drop-oldest. A queue would let a slow consumer
  build a backlog of stale frames — wrong for live navigation, where only the newest frame
  matters. A slow stage drops frames instead of stalling the stage upstream.

### 3D + vehicle geometry
- Depth enabled (`DEPTH_MODE.NEURAL`, `UNIT.METER`, `COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD`).
- `retrieve_measure(MEASURE.XYZRGBA)` gives a stereo-triangulated 3D point per pixel in the
  **LEFT lens optical frame** — this is where the two-lens stereo geometry + calibration is applied.
- The camera is physically offset from the vehicle centerline, so a target pixel is **not** the
  vehicle center. `MOUNT_TRANSLATION` / `MOUNT_ROTATION_RPY` transform LEFT-optical → vehicle frame.
  **These are PLACEHOLDER `[0,0,0]` — measure and fill them before trusting absolute position.**
- Intrinsics (`fx/fy/cx/cy`) read from `calibration_parameters.left_cam`, used to re-project the 3D
  path back onto the image.

### Coordinate frames
Both frames use **X forward, Y left, Z up** (metres). LEFT optical origin = left lens; vehicle
origin = centerline. Related by the mount transform above.

### Runtime target switching
Navigates toward one **active class** at a time (mission changes it over time):
- Keys in the window: `0`–`7` select class, `n`/`p` cycle, `q` quit.
- From code: `import object_detection; object_detection.set_target_class(id)`.
- Track coasts through brief dropouts (`TRACK_MAX_LOST_FRAMES`), EMA-smoothed (`TRACK_EMA_ALPHA`).

### HUD
Active class + status (`LOCKED` / `COASTING(lost n)` / `SEARCHING`), target `x/y/z` (m), distance,
yaw/pitch (deg). Nav arrow from bottom-center to target; orange polyline = projected 3D route.

## Dependencies

`test_scripts/object_detection.py` needs (versions = currently verified working set):

| Package | Version | Install | Note |
|---------|---------|---------|------|
| Python | 3.10.12 | — | |
| ZED SDK | 5.3.0 | Stereolabs installer | Hard prerequisite: provides the driver, depth, and `pyzed`. |
| `pyzed` | 5.3 | ZED SDK `get_python_api.py` | **Not pip.** Version must match the installed SDK + CUDA. |
| `onnxruntime-gpu` | 1.23.0 | `pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126` | **Must be the Jetson build.** Plain `onnxruntime` from PyPI is CPU-only. Do not install both. |
| CUDA / cuDNN / TensorRT | 12.6 / 9.3.0 / 10.3.0 | JetPack 6.2 (L4T R36.4.7) | Supplied by JetPack; matches the `jp6/cu126` wheel. |
| `tensorrt` (python) | 10.3.0 | JetPack 6.2 | Needed only for the native `.engine` backend. |
| `cuda-python` | 12.6.2 | `pip install cuda-python` | Device malloc / pinned memory / stream for the native backend. |
| `numpy` | 1.26.4 | `pip install numpy` | |
| `opencv-python` (`cv2`) | 4.12.0 | `pip install opencv-python` | Needs GUI support for `imshow` (headless build won't show the window). |

Versions are the current working set, not hard floors — treat ZED SDK ↔ `pyzed` ↔ CUDA as the
coupled trio; the pip packages are looser.

## Run

```bash
# live detector (needs ZED connected)
python3 test_scripts/object_detection.py

# offline geometry/tracker unit checks (no camera)
cd test_scripts && python3 test_geometry.py
```

```bash
# ignore the prebuilt engine and go through ONNX Runtime (CUDA EP)
USE_NATIVE_TRT=0 python3 test_scripts/object_detection.py

# opt into ORT's own TensorRT EP (first run builds an engine, ~8 min, idle board only)
USE_NATIVE_TRT=0 USE_TENSORRT=1 python3 test_scripts/object_detection.py
```

The backend is selected automatically and the startup banner states which one won. If it
says `CPU FALLBACK`, the wrong `onnxruntime` package is installed — the banner prints the
exact fix.

Rebuilding the engine for a different board:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=test_scripts/ffc_rs_26.onnx \
  --saveEngine=test_scripts/ffc_rs_26.engine --fp16
```

## Out of scope / TODO
- Dashboard: the 3D model's rendered orientation has **not** been visually confirmed against
  the physical vehicle. The axis mapping is correct by construction (quaternion +
  change-of-basis), but needs a human eye on the real sub.
- Dashboard is monitoring-only — no arm/move commands from the browser.
- Capture-stage cost (~36 ms) is the current FPS ceiling — see *Measured live performance*.
- The prebuilt engine's precision has not been diffed against fp32 ONNX output for accuracy.
- `ffc_rs_26.engine` is gitignored and board-specific — a new Jetson needs its own build.
- Obstacle avoidance / occupancy map — route is straight-line heading only.
- ROS2 topic / TF publishing — `object_detection.py` is standalone.
- Segmentation mask rendering (`output1` prototypes).
- Real mount-offset values (currently placeholder zeros).
