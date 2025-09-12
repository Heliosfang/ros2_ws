#!/usr/bin/env python3
import sys
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetPlan


class TestClient(Node):
    def __init__(self):
        super().__init__("test_client")

        # Declare simple params so you can tweak start/goal without editing code
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("goal_x",  10.0)
        self.declare_parameter("goal_y",  0.0)
        self.declare_parameter("tolerance", 0.0)
        self.declare_parameter("service_name", "get_path_from_txt")

        self.client = self.create_client(
            GetPlan, self.get_parameter("service_name").get_parameter_value().string_value
        )

        self.get_logger().info(f"Waiting for service '{self.client.srv_name}'...")
        if not self.client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Service not available after 10 seconds.")
            rclpy.shutdown()
            sys.exit(1)
        self.get_logger().info("Service available.")

        request = self._build_request()
        self.future = self.client.call_async(request)
        self.future.add_done_callback(self._handle_response)

    def _build_request(self) -> GetPlan.Request:
        p = self.get_parameters([
            "frame_id", "start_x", "start_y", "goal_x", "goal_y", "tolerance"
        ])
        P = {pp.name: pp.value for pp in p}

        req = GetPlan.Request()

        # Fill start
        start = PoseStamped()
        start.header.frame_id = P["frame_id"]
        start.pose.position.x = float(P["start_x"])
        start.pose.position.y = float(P["start_y"])
        start.pose.position.z = 0.0
        start.pose.orientation.w = 1.0
        req.start = start

        # Fill goal
        goal = PoseStamped()
        goal.header.frame_id = P["frame_id"]
        goal.pose.position.x = float(P["goal_x"])
        goal.pose.position.y = float(P["goal_y"])
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        req.goal = goal

        req.tolerance = float(P["tolerance"])
        return req

    def _handle_response(self, future: rclpy.task.Future):
        try:
            resp: Optional[GetPlan.Response] = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            rclpy.shutdown()
            return

        if resp is None:
            self.get_logger().error("No response received.")
            rclpy.shutdown()
            return

        path = resp.plan
        n = len(path.poses)
        self.get_logger().info(f"Received path with {n} poses (frame_id='{path.header.frame_id}').")

        # Print every pose (x, y, z)
        for i, ps in enumerate(path.poses):
            x = ps.pose.position.x
            y = ps.pose.position.y
            z = ps.pose.position.z
            print(f"{i:04d}: x={x:.6f}, y={y:.6f}, z={z:.6f}")

        # Exit after printing once
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TestClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
