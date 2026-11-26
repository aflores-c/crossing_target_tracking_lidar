import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'crossing_target_tracking'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'PAPER_REPRODUCTION.md']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andy Flores',
    maintainer_email='andy.flores@pucp.edu.pe',
    description='Two-level LiDAR extended-target tracking (Pieroni et al. 2024, Sec. 3).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_node = crossing_target_tracking.ros.tracking_node:main',
            'inspect_bag = crossing_target_tracking.tools.inspect_bag:main',
            'offline_process = crossing_target_tracking.tools.offline_process:main',
            'evaluate = crossing_target_tracking.tools.evaluate:main',
            'run_experiments = crossing_target_tracking.tools.run_experiments:main',
            'synthetic_experiments = crossing_target_tracking.tools.synthetic_experiments:main',
            'compare_playback = crossing_target_tracking.tools.compare_playback:main',
        ],
    },
)
