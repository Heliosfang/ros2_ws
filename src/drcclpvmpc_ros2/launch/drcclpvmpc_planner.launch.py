from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    cfg_file = os.path.join(os.getenv('HOME'), 'ros2_ws/src/drcclpvmpc_ros2/config/lincoln_planner.yaml')
    return LaunchDescription([
        Node(
            package="drcclpvmpc_ros2",
            executable="drcclpvmpc_main",
            name="drcclpvmpc_main",
            output="screen",
            remappings=[
                ("/odom", "/carla/odom"),
                ("/obs_box", "/car_box")
            ],
            parameters=[cfg_file]
        )
    ])
