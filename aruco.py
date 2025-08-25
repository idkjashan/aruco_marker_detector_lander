#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import TrajectorySetpoint, VehicleCommand
from sensor_msgs.msg import Image
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
import numpy as np

class DroneState:
    SEARCHING = 1
    CENTERING = 2
    ALIGNED = 3
    LANDING = 4

class ArucoLandNode(Node):
    def __init__(self):
        super().__init__('aruco_land_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
            
        self.image_sub = self.create_subscription(
            Image, '/world/aruco/model/x500_depth_0/link/camera_link/sensor/IMX214/image', self.image_callback, qos_profile)

        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.state = DroneState.SEARCHING
        self.bridge = CvBridge()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        
        self.target_marker_id = 0
        self.img_height = None
        self.img_width = None
        self.marker_center_x = None
        self.marker_center_y = None
        self.marker_detected = False
        
        self.k_p_forward = 0.006 
        self.k_p_sideways = 0.006 
        self.k_p_yaw = 0.0

        self.align_threshold_px = 20
        self.align_counter = 0
        self.align_frames_required = 20

        self.get_logger().info("ArUco Landing Node initialized with CORRECTED SIMPLE Kp control.")

    def image_callback(self, msg):
        """Processes camera images to find the ArUco marker."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.img_height is None:
                self.img_height, self.img_width, _ = cv_image.shape
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None and self.target_marker_id in ids:
                self.marker_detected = True
                index = np.where(ids == self.target_marker_id)[0][0]
                marker_corners = corners[index][0]
                
                self.marker_center_x = int(np.mean(marker_corners[:, 0]))
                self.marker_center_y = int(np.mean(marker_corners[:, 1]))

                aruco.drawDetectedMarkers(cv_image, corners, ids)
            else:
                self.marker_detected = False

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")
        
        cv2.imshow("Camera Feed", cv_image)
        cv2.waitKey(1)

    def control_loop(self):
        """Main state machine and control logic."""
        if self.state == DroneState.SEARCHING:
            self.get_logger().info("State: SEARCHING", throttle_duration_sec=1)
            self.publish_hover_setpoint()
            if self.marker_detected:
                self.state = DroneState.CENTERING
                self.get_logger().info("Marker detected. Transitioning to CENTERING.")

        elif self.state == DroneState.CENTERING:
            self.get_logger().info("State: CENTERING", throttle_duration_sec=1)

            if not self.marker_detected:
                self.get_logger().warn("Marker lost. Reverting to SEARCHING.")
                self.state = DroneState.SEARCHING
                self.publish_hover_setpoint()
                return

            img_center_x = self.img_width / 2
            img_center_y = self.img_height / 2
            
            error_x = img_center_x - self.marker_center_x
            error_y = img_center_y - self.marker_center_y
            

            px4_vel_x = self.k_p_forward * error_x

            px4_vel_y = self.k_p_sideways * error_y
            
            yawspeed = 0.0
            
            self.publish_velocity_setpoint(px4_vel_x, px4_vel_y, yawspeed=yawspeed)

            if abs(error_x) < self.align_threshold_px and abs(error_y) < self.align_threshold_px:
                self.align_counter += 1
                if self.align_counter >= self.align_frames_required:
                    self.state = DroneState.ALIGNED
                    self.get_logger().info("Drone is aligned. Transitioning to ALIGNED.")
            else:
                self.align_counter = 0

        elif self.state == DroneState.ALIGNED:
            self.get_logger().info("State: ALIGNED. Issuing land command.")
            self.publish_hover_setpoint()
            self.land()
            self.state = DroneState.LANDING

        elif self.state == DroneState.LANDING:
            self.get_logger().info("State: LANDING. Waiting for disarm.", throttle_duration_sec=1)
            pass

    def publish_hover_setpoint(self):
        """Publishes a zero-velocity setpoint to make the drone hover."""
        self.publish_velocity_setpoint(0.0, 0.0, yawspeed=0.0)

    def publish_velocity_setpoint(self, vx, vy, vz=0.0, yawspeed=np.nan):
        """Publishes a trajectory setpoint with specified NED velocities and yawspeed."""
        msg = TrajectorySetpoint()
        msg.position = [np.nan, np.nan, np.nan]
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.acceleration = [np.nan, np.nan, np.nan]
        msg.yaw = np.nan
        msg.yawspeed = float(yawspeed)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)
        self.get_logger().info(f"Published command: {command}")

    def land(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoLandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
