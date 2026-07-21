"""RViz visualization of vehicle state and the active motion command.

Publishes, at :data:`RvizPublisher.rate_hz`:

======================  ==========================  ==========================
topic                   type                        shows
======================  ==========================  ==========================
``/rov/odom``           ``nav_msgs/Odometry``       ZED fused pose + velocity
``/rov/imu``            ``sensor_msgs/Imu``         orientation, rates, accel
``/rov/heading``        ``visualization_msgs/``     heading arrow + compass ring
                        ``MarkerArray``
``/rov/command``        ``visualization_msgs/``     the demand being sent to
                        ``MarkerArray``             ArduSub right now
``/rov/goal``           ``visualization_msgs/``     where the active primitive
                        ``MarkerArray``             is driving to
``/rov/status``         ``std_msgs/String``         one-line HUD text
``/tf``                 ``odom`` -> ``base_link``   vehicle pose
======================  ==========================  ==========================

**Publish-only.** The bridge never commands motion, so RViz crashing or the
subscriber going away cannot affect the vehicle.

Colour convention across every marker: **cyan = measured** (what the vehicle
believes is true), **orange = commanded** (what it is doing about it),
**green = goal** (where it is trying to end up). The point of the display is
seeing measured and commanded disagree.

Run RViz on any machine on the same ROS_DOMAIN_ID::

    rviz2 -d rov/ros/rov.rviz
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from ..api.config import CONFIG, Config
from ..api.movement import MotionGoal, MotionState, Movement
from ..api.telemetry import Telemetry
from ..api.vehicle import Setpoint, Vehicle
from ..api.zed_pose import Pose, ZedPose

logger = logging.getLogger("rov.ros")

__all__ = ["RvizPublisher"]

#: Fixed frame; the odometry origin set by ``ZedPose.reset_odometry``.
ODOM_FRAME = "odom"
#: Vehicle body frame: X forward, Y left, Z up.
BASE_FRAME = "base_link"

# Marker colours (r, g, b, a).
CYAN = (0.0, 0.9, 1.0, 0.9)      # measured
ORANGE = (1.0, 0.55, 0.0, 0.9)   # commanded
GREEN = (0.2, 1.0, 0.3, 0.8)     # goal
RED = (1.0, 0.2, 0.2, 0.9)       # fault / limit
GREY = (0.7, 0.7, 0.7, 0.35)     # reference geometry

#: Metres of arrow per unit of normalized thrust. A full-scale command draws
#: a 1 m arrow, which is legible next to a vehicle roughly 0.5 m long.
THRUST_ARROW_SCALE = 1.0


class RvizPublisher:
    """Bridges the control stack onto ROS2 topics for RViz.

    Args:
        vehicle: Source of the live setpoint.
        zed: Source of pose and velocity.
        telemetry: Source of depth, attitude rates and battery.
        movement: Source of motion state and the active goal.
        config: Configuration bundle.
        rate_hz: Publication rate. 20 Hz is smooth in RViz and costs
            almost nothing next to the control and vision loops.

    Example:
        >>> publisher = RvizPublisher(vehicle, zed, telemetry, movement)
        >>> publisher.start()    # doctest: +SKIP
    """

    def __init__(self, vehicle: Vehicle, zed: ZedPose, telemetry: Telemetry,
                 movement: Movement, config: Config = CONFIG,
                 rate_hz: float = 20.0) -> None:
        self._vehicle = vehicle
        self._zed = zed
        self._telemetry = telemetry
        self._movement = movement
        self._cfg = config
        self.rate_hz = rate_hz

        self._node = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._owns_context = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Initialise ROS, create the node, and start publishing.

        Failure here is logged and swallowed: losing the display must never
        take down a running mission.
        """
        try:
            self._init_ros()
        except Exception:
            logger.exception("RViz bridge failed to start; continuing without it")
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="rov-rviz", daemon=True)
        self._thread.start()
        logger.info("RViz bridge publishing at %.0f Hz (fixed frame '%s')",
                    self.rate_hz, ODOM_FRAME)

    def stop(self) -> None:
        """Stop publishing and tear down the node. Idempotent."""
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # pragma: no cover
                logger.exception("failed to destroy ROS node")
            self._node = None
        if self._owns_context:
            import rclpy

            try:
                rclpy.shutdown()
            except Exception:  # pragma: no cover
                pass
            self._owns_context = False
        logger.info("RViz bridge stopped")

    def _init_ros(self) -> None:
        """Create the node and every publisher. Imports rclpy lazily."""
        import rclpy
        from geometry_msgs.msg import TransformStamped  # noqa: F401  (typing aid)
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        from std_msgs.msg import String
        from tf2_ros import TransformBroadcaster
        from visualization_msgs.msg import MarkerArray

        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True

        self._rclpy = rclpy
        self._node = rclpy.create_node("rov_visualizer")
        node = self._node

        # Depth 1 + keep-last: for a live display only the newest sample
        # matters, and a queue would just show RViz stale data after a hitch.
        self._odom_pub = node.create_publisher(Odometry, "/rov/odom", 1)
        self._imu_pub = node.create_publisher(Imu, "/rov/imu", 1)
        self._heading_pub = node.create_publisher(MarkerArray, "/rov/heading", 1)
        self._command_pub = node.create_publisher(MarkerArray, "/rov/command", 1)
        self._goal_pub = node.create_publisher(MarkerArray, "/rov/goal", 1)
        self._status_pub = node.create_publisher(String, "/rov/status", 1)
        self._tf = TransformBroadcaster(node)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        period = 1.0 / self.rate_hz
        next_tick = time.monotonic()
        while self._running.is_set():
            try:
                self._publish_once()
            except Exception:  # pragma: no cover - display must never kill the run
                logger.exception("RViz publish failed")
                time.sleep(0.2)
            next_tick += period
            sleep = next_tick - time.monotonic()
            time.sleep(sleep) if sleep > 0 else None
            if sleep <= 0:
                next_tick = time.monotonic()

    def _publish_once(self) -> None:
        """One frame of the display: read every source once, then publish."""
        pose = self._zed.get_pose()
        setpoint = self._vehicle.setpoint
        goal = self._movement.current_goal()
        state = self._movement.state
        stamp = self._node.get_clock().now().to_msg()

        self._publish_tf(pose, stamp)
        self._publish_odom(pose, stamp)
        self._publish_imu(pose, stamp)
        self._publish_heading(pose, stamp)
        self._publish_command(setpoint, state, stamp)
        self._publish_goal(goal, pose, stamp)
        self._publish_status(pose, setpoint, goal, state)

    # ------------------------------------------------------------------
    # Pose and IMU
    # ------------------------------------------------------------------
    def _publish_tf(self, pose: Pose, stamp) -> None:
        """Broadcast ``odom`` -> ``base_link``.

        Everything else can then be published in whichever frame is natural:
        body-frame markers (thrust demand) need no rotation maths, and RViz
        places them correctly from the transform.
        """
        from geometry_msgs.msg import TransformStamped

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = ODOM_FRAME
        transform.child_frame_id = BASE_FRAME
        transform.transform.translation.x = float(pose.x)
        transform.transform.translation.y = float(pose.y)
        transform.transform.translation.z = float(pose.z)
        qx, qy, qz, qw = _quaternion_from_rpy(pose.roll, pose.pitch, pose.yaw)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self._tf.sendTransform(transform)

    def _publish_odom(self, pose: Pose, stamp) -> None:
        """Fused pose plus body-frame twist."""
        from nav_msgs.msg import Odometry

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = ODOM_FRAME
        msg.child_frame_id = BASE_FRAME
        msg.pose.pose.position.x = float(pose.x)
        msg.pose.pose.position.y = float(pose.y)
        msg.pose.pose.position.z = float(pose.z)
        qx, qy, qz, qw = _quaternion_from_rpy(pose.roll, pose.pitch, pose.yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.twist.twist.linear.x = float(pose.vx)
        msg.twist.twist.linear.y = float(pose.vy)
        msg.twist.twist.linear.z = float(pose.vz)
        attitude = self._telemetry.get_attitude()
        if attitude is not None:
            msg.twist.twist.angular.x = math.radians(attitude.roll_rate)
            msg.twist.twist.angular.y = math.radians(attitude.pitch_rate)
            msg.twist.twist.angular.z = math.radians(attitude.yaw_rate)
        self._odom_pub.publish(msg)

    def _publish_imu(self, pose: Pose, stamp) -> None:
        """Inertial state, blended from the two sources that have it.

        Orientation comes from the ZED (vision-corrected, so it does not
        drift in yaw the way an unaided IMU does); angular rates come from
        the autopilot's EKF; linear acceleration from the ZED's IMU. RViz's
        Imu display shows the acceleration vector, which makes an impact or a
        thruster kick immediately visible.
        """
        from sensor_msgs.msg import Imu

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = BASE_FRAME
        qx, qy, qz, qw = _quaternion_from_rpy(pose.roll, pose.pitch, pose.yaw)
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        attitude = self._telemetry.get_attitude()
        if attitude is not None:
            msg.angular_velocity.x = math.radians(attitude.roll_rate)
            msg.angular_velocity.y = math.radians(attitude.pitch_rate)
            msg.angular_velocity.z = math.radians(attitude.yaw_rate)

        ax, ay, az = self._zed.get_acceleration()
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az

        # -1 in element 0 is the MAVLink/ROS convention for "no covariance
        # estimate available", which is honest: neither source reports one.
        msg.orientation_covariance[0] = -1.0 if not pose.valid else 0.0
        self._imu_pub.publish(msg)

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------
    def _publish_heading(self, pose: Pose, stamp) -> None:
        """Heading arrow and a compass ring, both in the body frame."""
        markers = [
            self._arrow("heading", 0, BASE_FRAME, stamp,
                        (0.0, 0.0, 0.0), (0.8, 0.0, 0.0),
                        CYAN if pose.valid else RED, shaft=0.03),
            self._ring("heading", 1, BASE_FRAME, stamp, radius=0.8, color=GREY),
            self._text("heading", 2, BASE_FRAME, stamp, (0.9, 0.0, 0.15),
                       f"yaw={pose.yaw:+.1f}", CYAN if pose.valid else RED,
                       size=0.11),
        ]
        self._publish_markers(self._heading_pub, markers)

    def _publish_command(self, setpoint: Setpoint, state: MotionState, stamp) -> None:
        """The demand currently going to ArduSub, drawn in the body frame.

        Body frame is the right choice here: a translation demand of
        "forward 0.3" is a statement about the vehicle, not the world, and
        drawing it in ``base_link`` means it stays pinned to the hull as the
        vehicle yaws — exactly how a pilot thinks about it.
        """
        translation = np.array([setpoint.forward, setpoint.strafe, setpoint.vertical])
        magnitude = float(np.linalg.norm(translation))
        color = ORANGE if state is not MotionState.ERROR else RED

        markers: List = []
        if magnitude > 1e-3:
            tip = tuple(translation * THRUST_ARROW_SCALE)
            markers.append(self._arrow("command", 0, BASE_FRAME, stamp,
                                       (0.0, 0.0, 0.0), tip, color, shaft=0.05))
        if abs(setpoint.yaw) > 1e-3:
            # Yaw is a rate demand, not a position, so it is drawn as an arc
            # whose length scales with the demand rather than as an arrow to
            # somewhere — there is no "somewhere" for a rate.
            markers.append(self._arc("command", 1, BASE_FRAME, stamp,
                                     radius=0.55,
                                     span_deg=setpoint.yaw * 120.0,
                                     color=color))
        # One value per line and no internal spaces: RViz renders a space in
        # marker text far wider than a character, which scatters a multi-column
        # HUD across the viewport.
        markers.append(self._text(
            "command", 2, BASE_FRAME, stamp, (0.0, 0.0, 0.45),
            f"{state.value}\n"
            f"fwd={setpoint.forward:+.2f}\n"
            f"str={setpoint.strafe:+.2f}\n"
            f"vert={setpoint.vertical:+.2f}\n"
            f"yaw={setpoint.yaw:+.2f}",
            color, size=0.09))
        self._publish_markers(self._command_pub, markers)

    def _publish_goal(self, goal: Optional[MotionGoal], pose: Pose, stamp) -> None:
        """Where the active primitive is driving to, in the odometry frame."""
        if goal is None:
            self._publish_markers(self._goal_pub, [])
            return

        markers: List = []
        if goal.position is not None:
            target = tuple(float(v) for v in goal.position)
            markers.append(self._sphere("goal", 0, ODOM_FRAME, stamp, target,
                                        diameter=2 * self._cfg.motion.position_tolerance
                                        * 10.0, color=GREEN))
            # The straight line is the *plan*, not the path taken; comparing
            # it against the odom trail in RViz is how cross-track error
            # becomes visible.
            markers.append(self._line("goal", 1, ODOM_FRAME, stamp,
                                      [(pose.x, pose.y, pose.z), target], GREEN))
            remaining = float(np.linalg.norm(goal.position - pose.position))
            markers.append(self._text("goal", 2, ODOM_FRAME, stamp,
                                      (target[0], target[1], target[2] + 0.2),
                                      f"{goal.kind}\n{remaining * 100:.1f}cm",
                                      GREEN, size=0.10))
        if goal.heading is not None:
            length = 0.9
            tip = (pose.x + length * math.cos(math.radians(goal.heading)),
                   pose.y + length * math.sin(math.radians(goal.heading)),
                   pose.z)
            markers.append(self._arrow("goal", 3, ODOM_FRAME, stamp,
                                       (pose.x, pose.y, pose.z), tip,
                                       GREEN, shaft=0.02))
        self._publish_markers(self._goal_pub, markers)

    def _publish_status(self, pose: Pose, setpoint: Setpoint,
                        goal: Optional[MotionGoal], state: MotionState) -> None:
        """One-line HUD for ``ros2 topic echo /rov/status``."""
        from std_msgs.msg import String

        depth = self._telemetry.get_depth()
        battery = self._telemetry.get_battery().voltage
        msg = String()
        msg.data = (
            f"state={state.value} goal={goal.kind if goal else '-'} "
            f"tracking={'OK' if self._zed.is_tracking() else 'LOST'} "
            f"pos=({pose.x:+.3f},{pose.y:+.3f},{pose.z:+.3f}) yaw={pose.yaw:+.1f} "
            f"depth={'n/a' if depth is None else f'{depth:.2f}'} "
            f"cmd=({setpoint.forward:+.2f},{setpoint.strafe:+.2f},"
            f"{setpoint.vertical:+.2f},{setpoint.yaw:+.2f}) "
            f"batt={'n/a' if battery is None else f'{battery:.1f}V'} "
            f"armed={self._vehicle.armed}"
        )
        self._status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Marker helpers
    # ------------------------------------------------------------------
    def _publish_markers(self, publisher, markers: List) -> None:
        """Send a marker array, clearing anything left from the last frame.

        RViz keeps markers until they expire or are deleted. A DELETEALL
        prefix means a marker that stops being published (the goal arrow when
        a move finishes) disappears immediately instead of lingering as a
        stale claim about what the vehicle is doing.
        """
        from visualization_msgs.msg import Marker, MarkerArray

        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        array.markers.extend(markers)
        publisher.publish(array)

    def _base_marker(self, namespace: str, marker_id: int, frame: str, stamp,
                     marker_type: int, color: Tuple[float, float, float, float]):
        from builtin_interfaces.msg import Duration
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.pose.orientation.w = 1.0
        # Outlive one publish period so the display does not flicker, but
        # expire quickly if publishing stops — a frozen marker is a lie.
        marker.lifetime = Duration(sec=0, nanosec=int(3e9 / self.rate_hz))
        return marker

    def _arrow(self, namespace: str, marker_id: int, frame: str, stamp,
               start: Tuple[float, float, float], end: Tuple[float, float, float],
               color, shaft: float = 0.03):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        marker = self._base_marker(namespace, marker_id, frame, stamp,
                                   Marker.ARROW, color)
        marker.points = [Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
                         Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))]
        marker.scale.x = shaft            # shaft diameter
        marker.scale.y = shaft * 2.0      # head diameter
        marker.scale.z = shaft * 2.5      # head length
        return marker

    def _sphere(self, namespace: str, marker_id: int, frame: str, stamp,
                center: Tuple[float, float, float], diameter: float, color):
        from visualization_msgs.msg import Marker

        marker = self._base_marker(namespace, marker_id, frame, stamp,
                                   Marker.SPHERE, color)
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = \
            (float(v) for v in center)
        # Floor the size so a 8 mm tolerance sphere is still clickable in RViz.
        size = max(0.08, diameter)
        marker.scale.x = marker.scale.y = marker.scale.z = size
        return marker

    def _line(self, namespace: str, marker_id: int, frame: str, stamp,
              points, color, width: float = 0.015):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        marker = self._base_marker(namespace, marker_id, frame, stamp,
                                   Marker.LINE_STRIP, color)
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                         for p in points]
        marker.scale.x = width
        return marker

    def _ring(self, namespace: str, marker_id: int, frame: str, stamp,
              radius: float, color, segments: int = 48):
        """Reference circle, so heading is readable without reading numbers."""
        points = [(radius * math.cos(t), radius * math.sin(t), 0.0)
                  for t in np.linspace(0.0, 2 * math.pi, segments)]
        return self._line(namespace, marker_id, frame, stamp, points, color,
                          width=0.006)

    def _arc(self, namespace: str, marker_id: int, frame: str, stamp,
             radius: float, span_deg: float, color, segments: int = 24):
        """Arc from +X, spanning ``span_deg`` (signed: positive is to port)."""
        span = math.radians(span_deg)
        points = [(radius * math.cos(t), radius * math.sin(t), 0.0)
                  for t in np.linspace(0.0, span, segments)]
        return self._line(namespace, marker_id, frame, stamp, points, color,
                          width=0.02)

    def _text(self, namespace: str, marker_id: int, frame: str, stamp,
              position: Tuple[float, float, float], text: str, color,
              size: float = 0.1):
        from visualization_msgs.msg import Marker

        marker = self._base_marker(namespace, marker_id, frame, stamp,
                                   Marker.TEXT_VIEW_FACING, color)
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = \
            (float(v) for v in position)
        marker.scale.z = size
        marker.text = text
        return marker


def _quaternion_from_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float
                         ) -> Tuple[float, float, float, float]:
    """Convert roll/pitch/yaw in degrees to a ``(x, y, z, w)`` quaternion.

    Z-Y-X intrinsic order, matching both the ROS convention and the mount
    transform in :mod:`rov.api.zed_pose`, so the RViz axes line up with what
    the controller thinks the vehicle is doing.
    """
    roll, pitch, yaw = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)
