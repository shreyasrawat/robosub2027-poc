"""ROS2 bridge.

Optional. Nothing in :mod:`rov.api` imports this package, and nothing here is
required for autonomy — it exists so a human can watch what the vehicle
believes and what it is about to do, in RViz, while a mission runs.

The bridge is *publish-only by design*. It never commands motion, so a
crashed or paused RViz cannot influence the vehicle, and the mission behaves
identically whether or not anyone is watching.

Import is lazy: ``rclpy`` is only loaded when :class:`RvizPublisher` is
constructed, so the control stack still runs on a machine with no ROS
installed.
"""

from .visualizer import RvizPublisher

__all__ = ["RvizPublisher"]
