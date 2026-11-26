"""Run the crossing_target_tracking node (live or with an externally played bag).

  ros2 launch crossing_target_tracking tracking.launch.py \
      [config:=<yaml>] [use_sim_time:=true] [rviz:=true] [overrides:="['shadow.enabled=false']"]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('crossing_target_tracking')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(share, 'config', 'crossing_tracking.yaml')),
        DeclareLaunchArgument('overrides', default_value="['']",
                              description="YAML list of 'section.key=value' overrides"),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        Node(package='crossing_target_tracking', executable='tracking_node', name='crossing_target_tracking',
             output='screen',
             parameters=[{'config_file': LaunchConfiguration('config'),
                          'overrides': LaunchConfiguration('overrides'),
                          'use_sim_time': use_sim_time}],
             additional_env={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}),
        Node(package='rviz2', executable='rviz2', name='crossing_tracking_rviz', output='log',
             arguments=['-d', os.path.join(share, 'rviz', 'crossing_tracking.rviz')],
             parameters=[{'use_sim_time': use_sim_time}],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
