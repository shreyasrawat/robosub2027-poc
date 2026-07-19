# RoboSub 2027 Workspace

ROS2 (colcon) workspace for the RoboSub 2027 autonomous underwater vehicle.

## Layout

| Path | What |
|------|------|
| `src/zed-ros2-wrapper/` | Vendored Stereolabs ZED ROS2 wrapper (C++). Third-party; own git repo. |
| `test_scripts/` | First-party code. Standalone Python (no ROS runtime). |
| `plans/` | Plan / design documents for future work. |
| `build/` `install/` `log/` | colcon artifacts. Do not edit. |

Only `test_scripts/` and `plans/` and this file are first-party. Everything under `src/` is vendored.

## `test_scripts/object_detection.py`

Standalone ZED stereo detector + 3D target tracker + navigation-path overlay.

**Per-frame pipeline:** grab → retrieve LEFT image + XYZ point cloud → ONNX detect →
sample target's 3D point → transform to vehicle frame → track active class →
straight-line path → render boxes + nav arrow + path line + HUD.

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
| `onnxruntime` / `onnxruntime-gpu` | 1.23.2 | `pip install onnxruntime-gpu` | GPU build on the Jetson for the CUDA provider; CPU build elsewhere. |
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

On the Jetson the CUDA ONNX provider is used automatically; elsewhere it falls back to CPU
(harmless warning).

## Out of scope / TODO
- Obstacle avoidance / occupancy map — route is straight-line heading only.
- ROS2 topic / TF publishing — `object_detection.py` is standalone.
- Segmentation mask rendering (`output1` prototypes).
- Real mount-offset values (currently placeholder zeros).
