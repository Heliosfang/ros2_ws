from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    cfg_file = os.path.join(get_package_share_directory('drcclpvmpc_ros2'), 'config', 'lincoln_planner.yaml')
    # home = os.getenv('HOME')
    # cfg_file = os.path.join(home, 'ros2_ws/src/drcclpvmpc_ros2/config/lincoln_planner.yaml')

    # Optional: point to an RViz config if you have one
    # rviz_cfg = os.path.join(home, 'ros2_ws/src/drcclpvmpc_ros2/config/planner_view.rviz')

    return LaunchDescription([
        # Your planner node
        Node(
            package="drcclpvmpc_ros2",
            executable="drcclpvmpc_main",
            name="drcclpvmpc_main",
            output="screen",
            remappings=[
                ("/odom", "/carla/odom"),
                ("/obs_box", "/car_box"),
            ],
            parameters=[cfg_file],
        ),
    ])
