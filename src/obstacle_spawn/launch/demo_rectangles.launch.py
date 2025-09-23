from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('obstacle_spawn')
    rviz_cfg = os.path.join(pkg, 'config', 'costmap_rectangles.rviz')

    use_rviz = DeclareLaunchArgument('use_rviz', default_value='true')

    nodes = [
        Node(
            package='obstacle_spawn',                     
            executable='dummy_costmap_pub',           
            name='dummy_costmap_pub',
            output='screen'
        ),
        Node(
            package='obstacle_spawn',
            executable='costmap_to_rectangles',       
            name='costmap_to_rectangles',
            output='screen',
            parameters=[{
                'costmap_topic': '/local_costmap/costmap',
                'occupied_threshold': 50,
                'min_cluster_cells': 6,
                'max_rectangles': 200,
                'publish_rate': 5.0,
                'frame_id': ''   # use costmap's frame ("map")
            }]
        ),
    ]

    # Optional RViz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        output='screen',
        condition=None  
    )

    return LaunchDescription([use_rviz] + nodes + [rviz])
