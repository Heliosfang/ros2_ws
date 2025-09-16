#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import open3d as o3d
import numpy as np

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class PCDPublisher(Node):
    def __init__(self):
        super().__init__('pcd_publisher')

        # --- Load PCD file ---
        home_dir = os.path.expanduser("~")
        pcd_file = os.path.join(home_dir, "Carla-0916", "HDMaps", "Town05.pcd")
        self.get_logger().info(f"Loading PCD: {pcd_file}")

        pcd = o3d.io.read_point_cloud(pcd_file)

        # --- Downsample to reduce number of points ---
        voxel_size = 0.5  # meters, adjust as needed
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        self.get_logger().info(f"Downsampled to {len(pcd.points)} points")

        pts = np.asarray(pcd.points, dtype=np.float32)

        # --- Convert to PointCloud2 ---
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "pcd_map"
        cloud_msg = point_cloud2.create_cloud_xyz32(header, pts)

        # --- Publisher (latched QoS) ---
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(PointCloud2, "pcd_points", qos)
        self.pub.publish(cloud_msg)
        self.get_logger().info("Published PCD to /pcd_points with frame_id=pcd_map")

        # --- Static transform map -> pcd_map ---
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "pcd_map"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info("Published static transform map -> pcd_map")


def main():
    rclpy.init()
    node = PCDPublisher()
    rclpy.spin(node)   # keeps TF alive
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
