from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Locate YAML in the share directory
    pkg_share = os.path.join(os.getenv('HOME'), 'ros2_ws/src/obstacle_spawn/config/obstacles.yaml')

    return LaunchDescription([
        Node(
            package='obstacle_spawn',
            executable='spawn_and_publish',
            name='obstacle_spawner',
            output='screen',
            parameters=[pkg_share],  # <-- load YAML here
        )
    ])
