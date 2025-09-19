from setuptools import find_packages, setup

package_name = 'drcclpvmpc_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/drcclpvmpc_planner.launch.py']),
        ('share/' + package_name + '/config', ['config/lincoln_planner.yaml']),
        ('share/' + package_name + '/config', ['config/planner_view.rviz']),
        ('share/' + package_name + '/meshes', ['meshes/Audi R8.dae']),
        ('share/' + package_name + '/meshes', ['meshes/Audi_R8_2017.stl']),
        ('share/' + package_name + '/meshes', ['meshes/DeLorean.STL']),
        
        ('lib/' + 'python3.12/' + 'site-packages/' + package_name + '/mpc' +
         '/data', [package_name + '/mpc/data/model_error.csv']),
        ('lib/' + 'python3.12/' + 'site-packages/' + package_name + '/mpc' +
         '/data', [package_name + '/mpc/data/z_states.csv']),
        ('lib/' + 'python3.12/' + 'site-packages/' + package_name + '/mpc' +
         '/data', [package_name + '/mpc/data/noise_arr.npy']),
        ('lib/' + 'python3.12/' + 'site-packages/' + package_name + '/mpc' +
         '/data', [package_name + '/mpc/data/noise_arrvx.npy']),
        ('lib/' + 'python3.12/' + 'site-packages/' + package_name + '/mpc' +
         '/output', [package_name + '/mpc/output/DynamicDRO_solver_output.txt']),
        ('lib/' + 'python3.12/' + 'site-packages/' + 'meshes', ['meshes/Audi R8.dae']),
        ('lib/' + 'python3.12/' + 'site-packages/' + 'meshes', ['meshes/Audi_R8_2017.stl']),
        ('lib/' + 'python3.12/' + 'site-packages/' + 'meshes', ['meshes/DeLorean.STL']),
        
        
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alion',
    maintainer_email='alion@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drcclpvmpc_main = drcclpvmpc_ros2.drcclpvmpc_main:main',
        ],
    },
)
