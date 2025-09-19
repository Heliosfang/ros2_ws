from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Locate YAML in the share directory
    pkg_share = os.path.join(get_package_share_directory('obstacle_spawn'), 'config', 'static_obstacles.yaml')

    return LaunchDescription([
        Node(
            package='obstacle_spawn',
            executable='static_obstacle_spawn',
            name='static_obstacle_spawner',
            output='screen',
            parameters=[pkg_share],  # <-- load YAML here
        )
    ])
