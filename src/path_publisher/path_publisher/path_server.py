#!/usr/bin/env python3
import os
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan


def load_xy_txt(txt_path: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            a = s.replace(",", " ").split()
            if len(a) < 2:
                continue
            pts.append((float(a[0]), float(a[1])))
    if len(pts) < 2:
        raise ValueError(f"Need at least 2 points in '{txt_path}', got {len(pts)}")
    return pts


class PathFromTxtService(Node):
    def __init__(self):
        super().__init__("path_from_txt_service")

        # Try installed share/output first, then source tree output
        candidates = []
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory("path_publisher")
            candidates.append(os.path.join(share, "output", "bspline.txt"))
        except Exception:
            pass
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        pkg_root_src = os.path.abspath(os.path.join(script_dir, ".."))
        candidates.append(os.path.join(pkg_root_src, "output", "bspline.txt"))

        self.txt_path = next((p for p in candidates if os.path.exists(p)), None)
        if not self.txt_path:
            raise FileNotFoundError(
                "bspline.txt not found. Looked in:\n  " + "\n  ".join(candidates) +
                "\nRun the generator first: path_publisher/path_publisher/bspline_generate.py"
            )

        # Preload and build a Path message once
        xy = load_xy_txt(self.txt_path)

        path = Path()
        path.header.frame_id = "map"
        for x, y in xy:
            ps = PoseStamped()
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self._path_template = path

        # Advertise service
        self._srv = self.create_service(GetPlan, "get_path_from_txt", self._handle_get_plan)

        self.get_logger().info(
            f"Service 'get_path_from_txt' ready. Loaded {len(self._path_template.poses)} poses from '{self.txt_path}'."
        )

    def _handle_get_plan(self, request: GetPlan.Request, response: GetPlan.Response) -> GetPlan.Response:
        # Stamp & frame the response freshly on each call
        now = self.get_clock().now().to_msg()

        path = Path()
        path.header.stamp = now
        path.header.frame_id = self._path_template.header.frame_id

        for src in self._path_template.poses:
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = path.header.frame_id
            ps.pose = src.pose
            path.poses.append(ps)

        response.plan = path

        # Optional log — shows goal received & count returned
        goal = request.goal.pose.position
        self.get_logger().info(
            f"Request received (goal ~ [{goal.x:.2f}, {goal.y:.2f}]). "
            f"Returning path with {len(path.poses)} poses."
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PathFromTxtService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
