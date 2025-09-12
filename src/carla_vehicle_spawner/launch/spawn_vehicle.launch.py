# launch/control_stack.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    cfg = os.path.join(os.getenv('HOME'), 'ros2_ws/src/carla_vehicle_spawner/config/params.yaml')
    return LaunchDescription([
        Node(package='carla_vehicle_spawner', executable='spawn_and_publish',
             name='carla_vehicle_spawner', output='screen', parameters=[cfg]),
        Node(package='carla_vehicle_spawner', executable='odom_publisher',
             name='carla_odom_publisher', output='screen', parameters=[cfg]),
        Node(package='carla_vehicle_spawner', executable='vehicle_controller',
             name='carla_vehicle_controller', output='screen', parameters=[cfg]),
    ])
