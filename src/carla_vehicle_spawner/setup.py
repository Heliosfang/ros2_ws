from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'carla_vehicle_spawner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # ✅ Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        # ✅ Install config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        ('share/' + package_name + '/config', ['config/planner_view.rviz']),  # optional
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Spawn a vehicle in CARLA and publish LiDAR/Camera data',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'pcd_loader = carla_vehicle_spawner.pcd_loader:main',
        'carla_camera_attacher = carla_vehicle_spawner.carla_camera_attacher:main',
        'lidar_sensor = carla_vehicle_spawner.lidar_sensor:main',
        'car_and_spectator = carla_vehicle_spawner.car_and_spectator:main',
        'odom_publisher = carla_vehicle_spawner.odom_publisher:main',
        'vehicle_controller = carla_vehicle_spawner.vehicle_controller:main',
        'vehicle_controller_keyboard = carla_vehicle_spawner.vehicle_controller_keyboard:main',
    ],
    },
)
