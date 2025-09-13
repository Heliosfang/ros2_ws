from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    home = os.getenv('HOME')
    cfg_file = os.path.join(home, 'ros2_ws/src/drcclpvmpc_ros2/config/lincoln_planner.yaml')

    # Optional: point to an RViz config if you have one
    rviz_cfg = os.path.join(home, 'ros2_ws/src/drcclpvmpc_ros2/config/rviz/planner_view.rviz')

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

        # Publish a static transform: map -> odom (identity)
        # If you prefer map->base_link, change the last two frame names accordingly.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom",
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
            output="screen",
        ),

        # RViz2 (loads a config if it exists; otherwise RViz starts with defaults)
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            # If you don't have an RViz config yet, you can remove the '-d' and path.
            arguments=["-d", rviz_cfg],
            output="screen",
        ),
    ])
