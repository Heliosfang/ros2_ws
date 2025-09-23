from setuptools import setup

package_name = 'obstacle_spawn'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/spawn_and_publish.launch.py']),
        ('share/' + package_name + '/launch', ['launch/static_spawn.launch.py']),
        ('share/' + package_name + '/launch', ['launch/demo_rectangles.launch.py']),
        ('share/' + package_name + '/config', ['config/obstacles.yaml']),
        ('share/' + package_name + '/config', ['config/static_obstacles.yaml']),
        ('share/' + package_name + '/config', ['config/costmap_rectangles.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Spawn two CARLA vehicles and publish their 2D bounding boxes.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'spawn_and_publish = obstacle_spawn.spawn_and_publish:main',
            'static_obstacle_spawn = obstacle_spawn.static_obstacle_spawn:main',
            'costmap_to_rectangles = obstacle_spawn.costmap_to_rectangles:main',
            'dummy_costmap_pub = obstacle_spawn.dummy_costmap_pub:main',
        ],
    },
)
