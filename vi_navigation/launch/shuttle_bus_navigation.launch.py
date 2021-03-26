#!/usr/bin/env python3
#
# Authors: Malte R. Damgaard

"""
Launches Gazebo and spawns a turtlebot3
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node

ROBOTNAMESPACE = os.environ['ROBOTNAMESPACE']
DOCKERNAMESPACE = os.environ['DOCKERNAMESPACE']


def generate_launch_description():
    velPub = Node(package='vi_navigation',
                  executable='shuttle_bus_navigation',
                  namespace=ROBOTNAMESPACE + '/' + DOCKERNAMESPACE,
                  remappings=[('/' + ROBOTNAMESPACE + '/' + DOCKERNAMESPACE + '/publishedVel',
                               '/' + ROBOTNAMESPACE + '/minimal' + '/cmd_vel'),
                              ('/' + ROBOTNAMESPACE + '/' + DOCKERNAMESPACE + '/odom',
                               '/' + ROBOTNAMESPACE + '/minimal' + '/odom')]
                  )

    return LaunchDescription([
        velPub,
    ])
