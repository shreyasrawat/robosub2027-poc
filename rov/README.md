# `rov/` — autonomous ROV control framework

High-level API over an ArduSub vehicle (Pixhawk) with a ZED 2i for
localization. Mission code talks to this package and nothing else — it never
imports PyMAVLink or the ZED SDK, and never sees a PWM value or a PID gain.

The Pixhawk keeps every low-level responsibility: stabilization, motor
mixing, failsafes, thruster outputs. This package only sends pilot-equivalent
`MANUAL_CONTROL` demands. **Nothing here bypasses the motor mixer.**

## Layers

Each layer may import only the ones above it in this table.

| Module | Responsibility | Exclusive dependency |
|---|---|---|
| `api/config.py` | every tunable; no logic | — |
| `api/controllers.py` | PID math | — |
| `api/vehicle.py` | MAVLink link, arming, modes, `MANUAL_CONTROL` | **only** pymavlink importer |
| `api/zed_pose.py` | visual-inertial pose, velocity, tracking health | **only** ZED SDK importer |
| `api/telemetry.py` | depth, attitude, battery, heartbeat | subscribes to `vehicle` |
| `api/movement.py` | closed-loop motion primitives | the mission-facing API |
| `api/mission.py` | behaviours composed from primitives | — |
| `vision/` | detections → normalized control targets | wraps `test_scripts/object_detection.py` |
| `ros/` | optional RViz bridge, publish-only | only `rclpy` importer |
| `rov.py` | the `ROV` facade: construction and shutdown order | — |

## Use

```python
from rov import ROV

with ROV(enable_vision=True, target_class=1) as rov:
    rov.set_mode("DEPTH_HOLD")
    rov.arm()
    rov.reset_odometry()

    rov.move_forward(0.05)      # 5 cm, closed-loop on ZED odometry
    rov.turn(90)                # degrees, positive to port
    rov.approach_target(distance_m=1.0)
    rov.hold_depth(1.0, duration=5)

    rov.disarm()
```

```bash
python3 -m rov.main --status                       # telemetry only, never arms
python3 -m rov.main --mission square --side 0.5
python3 -m rov.main --mission gate --vision --target-class 1
python3 -m rov.tests.test_api                      # offline math checks
```

## RViz visualization

```bash
# terminal 1 — vehicle, publishes while the mission runs
python3 -m rov.main --status --rviz
python3 -m rov.main --mission square --side 0.5 --rviz

# terminal 2 — viewer (any machine on the same ROS_DOMAIN_ID)
rviz2 -d rov/ros/rov.rviz
```

| topic | type | shows |
|---|---|---|
| `/tf` | `odom` → `base_link` | vehicle pose |
| `/rov/odom` | `nav_msgs/Odometry` | fused pose + velocity, breadcrumb trail |
| `/rov/imu` | `sensor_msgs/Imu` | orientation, body rates, acceleration |
| `/rov/heading` | `MarkerArray` | heading arrow, compass ring, yaw readout |
| `/rov/command` | `MarkerArray` | the demand going to ArduSub right now |
| `/rov/goal` | `MarkerArray` | where the active primitive is driving |
| `/rov/status` | `std_msgs/String` | one-line HUD for `ros2 topic echo` |

Colour convention: **cyan = measured**, **orange = commanded**, **green = goal**.
Watching measured and commanded disagree is the point of the display.

The bridge is **publish-only** — it never commands motion, so RViz crashing
cannot affect the vehicle, and a mission behaves identically whether or not
anyone is watching. `rclpy` is imported lazily, so the control stack still
runs on a machine with no ROS installed; if the bridge fails to start it logs
and the mission continues.

Two things the config gets right that are easy to get wrong: the IMU display
is `rviz_imu_plugin/Imu` (`rviz_default_plugins` has no Imu display), and its
acceleration scale is 0.01 because the ZED reports acceleration *including*
gravity — at the default scale a 9.8 m/s² column swamps the entire scene.

## Design decisions worth knowing

**Everything is closed-loop.** No primitive moves for a fixed duration or
sleeps its way to a destination. `move_forward(0.05)` computes a world-frame
waypoint, then runs PID against ZED odometry until the residual has stayed
inside tolerance for `settle_time`. An open-loop version is unrepeatable the
moment current, trim, or payload changes.

**World-frame waypoints, body-frame control.** A translation freezes the
heading and converts the requested body displacement into a world point.
If the vehicle yaws mid-move it still converges on the same physical spot
rather than chasing a rotating goal.

**Approach profile.** Inside `slowdown_distance` the output ceiling falls
linearly with remaining error, so the vehicle arrives slowly instead of
braking hard and overshooting. Outside `position_tolerance` the magnitude is
raised to `min_translation_output`, because a taper that reaches zero leaves
the vehicle stalled just short of the target. Those two rules together are
what make a 5 cm command land at 5 cm.

**One MAVLink reader.** pymavlink connections are not safe to read
concurrently; a second consumer silently steals messages from the first.
`Vehicle` owns the only receive loop and fans messages out to handlers.
`Telemetry` subscribes rather than opening its own link.

**One camera handle.** The ZED SDK allows one handle per device per process
and one grabbing thread. `ZedPose` opens and grabs; `DetectorService` borrows
the handle and only retrieves.

**Sender thread, not per-command sends.** ArduSub treats a gap in the pilot
stream as a failsafe, so `MANUAL_CONTROL` is republished at `rate_hz`
regardless of whether the setpoint changed. If no control loop refreshes the
setpoint within `setpoint_timeout` the sender reverts to neutral on its own —
a wedged control thread must not leave thrusters latched on.

**Depth comes from the barometer, position from the ZED.** Pressure stays
true over a long mission; visual odometry drifts vertically. `hold_depth()`
uses telemetry; `move_up`/`move_down` use ZED z for short relative moves.

**Heading comes from the ZED.** The autopilot's magnetometer sits inside a
hull with six thrusters. ZED yaw is primary, `telemetry.get_heading()` is the
cross-check and fallback.

**State machine.** `MotionState` serialises access — a second concurrent
command raises `MotionBusy` instead of fighting the first on the same
thrusters. A safety abort parks the controller in `ERROR` until
`clear_error()` is called deliberately.

**Safety checks run every control tick**: emergency-stop latch, heartbeat
age, ZED tracking health, depth envelope, battery. Any failure raises
`SafetyAbort`, and the thrusters are neutralised on every exit path including
exceptions.

## Configuration

All tunables live in `api/config.py`. Nothing else in the package defines a
number an operator would want to change. Override per run without editing the
file:

```python
import dataclasses
from rov import CONFIG, ROV

CONFIG.motion.position_tolerance = 0.005     # 5 mm
CONFIG.gains.heading.kp = 0.04
rov = ROV(CONFIG)
```

Starting gains are deliberately conservative — retune in the pool.

## Known gaps

- Mount offsets in `ZedConfig` are placeholder zeros, mirroring
  `test_scripts/object_detection.py`. Measure them before trusting absolute
  position.
- PID gains are untuned starting values; no wet testing has been done.
- No obstacle avoidance — `goto` is a straight line.
- No ROS2 publishing; this package is standalone like the detector.
