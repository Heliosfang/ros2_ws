#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose

import numpy as np
import math


class DummyCostmapPub(Node):
    def __init__(self):
        super().__init__('dummy_costmap_pub')
        self.pub = self.create_publisher(OccupancyGrid, '/local_costmap/costmap', 1)
        self.timer = self.create_timer(1.0, self.publish_costmap)  # publish every 1s
        self.get_logger().info("Publishing fake costmap on /local_costmap/costmap")

    def publish_costmap(self):
        width, height = 50, 50       # cells
        res = 0.1                    # 10 cm per cell
        ox, oy = 0.0, 0.0            # origin at (0,0)

        # start with free grid (-1=unknown, 0=free, 100=occupied)
        grid = np.zeros((height, width), dtype=np.int8)

        # add a block obstacle (rectangle cluster)
        grid[20:30, 10:20] = 100

        # add a rotated block (simulate angled obstacle)
        cx, cy = 35, 35
        for dx in range(-5, 6):
            for dy in range(-2, 3):
                x = int(cx + dx*math.cos(math.radians(30)) - dy*math.sin(math.radians(30)))
                y = int(cy + dx*math.sin(math.radians(30)) + dy*math.cos(math.radians(30)))
                if 0 <= x < width and 0 <= y < height:
                    grid[y, x] = 100

        # fill OccupancyGrid message
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = res
        msg.info.width = width
        msg.info.height = height
        msg.info.origin = Pose()
        msg.info.origin.position.x = ox
        msg.info.origin.position.y = oy
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = grid.flatten().tolist()

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = DummyCostmapPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
