from setuptools import find_packages, setup

package_name = 'path_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/output', []),   # ensure output/ exists in install space
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
        'path_server = path_publisher.path_server:main',
        'bspline_generate = path_publisher.bspline_generate:main',
        'test_client = path_publisher.test_client:main',
    ],
    },
)
