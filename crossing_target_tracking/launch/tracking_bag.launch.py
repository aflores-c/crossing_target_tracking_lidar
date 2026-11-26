"""Replay a rosbag2 with /clock and run the tracker on it (Mode A).

  ros2 launch crossing_target_tracking tracking_bag.launch.py bag:=<BAG_DIR> \
      [start_offset:=40] [rate:=1.0] [rviz:=true] [config:=<yaml>] [overrides:="[...]"]

Only the LiDAR and odometry topics are replayed (camera topics are not needed).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory('crossing_target_tracking')
    play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag'), '--clock', '100',
             '--rate', LaunchConfiguration('rate'), '--start-offset', LaunchConfiguration('start_offset'),
             '--topics', LaunchConfiguration('cloud_topic'), LaunchConfiguration('odom_topic')],
        output='screen')
    return LaunchDescription([
        DeclareLaunchArgument('bag'),
        DeclareLaunchArgument('start_offset', default_value='0.0'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('cloud_topic', default_value='/velodyne_points'),
        DeclareLaunchArgument('odom_topic', default_value='/utlidar/robot_odom'),
        DeclareLaunchArgument('config', default_value=os.path.join(share, 'config', 'crossing_tracking.yaml')),
        DeclareLaunchArgument('overrides', default_value="['']"),
        DeclareLaunchArgument('rviz', default_value='true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'tracking.launch.py')),
            launch_arguments={'config': LaunchConfiguration('config'),
                              'overrides': LaunchConfiguration('overrides'),
                              'use_sim_time': 'true', 'rviz': LaunchConfiguration('rviz')}.items()),
        TimerAction(period=2.0, actions=[play]),   # let the node subscribe first
    ])
