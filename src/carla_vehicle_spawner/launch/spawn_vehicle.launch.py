# launch/control_stack.launch.py
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    spawn_obstacle_package = get_package_share_directory('obstacle_spawn')
    vehicle_spawner_package = get_package_share_directory('carla_vehicle_spawner')
    cfg = os.path.join(vehicle_spawner_package, 'config', 'params.yaml')
    path_server_package = get_package_share_directory('path_publisher')
    return LaunchDescription([
        Node(package='carla_vehicle_spawner', executable='spawn_and_publish',
             name='carla_vehicle_spawner', output='screen', parameters=[cfg]),
        Node(package='carla_vehicle_spawner', executable='odom_publisher',
             name='carla_odom_publisher', output='screen', parameters=[cfg]),
        Node(package='carla_vehicle_spawner', executable='vehicle_controller',
             name='carla_vehicle_controller', output='screen', parameters=[cfg]),
        
        IncludeLaunchDescription(
               PythonLaunchDescriptionSource(
                    os.path.join(spawn_obstacle_package, 'launch', 'spawn_and_publish.launch.py')
               )
          ),
        Node(package='path_publisher', executable='path_server',
             name='path_server', output='screen'),
    ])
