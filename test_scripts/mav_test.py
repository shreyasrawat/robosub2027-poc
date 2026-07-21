from pymavlink import mavutil
import time

master = mavutil.mavlink_connection(
    "/dev/ttyACM0",
    baud=115200
)

master.wait_heartbeat()

print("Connected!")

master.arducopter_arm()

master.mav.manual_control_send(
    master.target_system,

    500,   # x (forward/back)
    0,     # y (strafe)
    500,   # z (throttle)
    0,     # r (yaw)

    0
)

time.sleep(2)

master.mav.manual_control_send(
    master.target_system,

    0,   # x (forward/back)
    0,     # y (strafe)
    0,   # z (throttle)
    0,     # r (yaw)

    0
)


master.arducopter_disarm()