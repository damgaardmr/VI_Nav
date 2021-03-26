#!/usr/bin/env python3
#
# Authors: Malte R. Damgaard

"""
Launches Gazebo and spawns a turtlebot3
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    ROBOTNAMESPACE = os.environ['ROBOTNAMESPACE']
    DOCKERNAMESPACE = os.environ['DOCKERNAMESPACE']

    SIMULATION = 'false'
    if 'SIMULATION' in os.environ:
        SIMULATION = os.environ['SIMULATION']
        use_sim_time = LaunchConfiguration('use_sim_time', default='SIMULATION')
        print("Simulation: " + SIMULATION)
    else:
        use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    velPub = Node(package='vi_navigation',
                  executable='shuttle_bus_navigation_cooperative',
                  parameters=[{'use_sim_time': use_sim_time}],
                  namespace=ROBOTNAMESPACE,  # + '/' + DOCKERNAMESPACE,
                  remappings=[('/' + ROBOTNAMESPACE + '/publishedVel',
                               '/' + ROBOTNAMESPACE + '/cmd_vel')]
                  # remappings=[('/' + ROBOTNAMESPACE + '/' + DOCKERNAMESPACE + '/publishedVel',
                  #             '/' + ROBOTNAMESPACE + '/minimal' + '/cmd_vel'),
                  #            ('/' + ROBOTNAMESPACE + '/' + DOCKERNAMESPACE + '/odom',
                  #             '/' + ROBOTNAMESPACE + '/minimal' + '/odom')]
                  )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',
                              default_value=SIMULATION,
                              description='Use simulation (Gazebo) clock if true'),
        velPub,
    ])
