#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:12:30 2021

@author: federico
"""

# =============================================================================
# En este archivo pongo funciones referidas a los ploteos que se hacen para
# observar los resultados de la simulación
# =============================================================================

from diffRobot import DiffRobot

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
import matplotlib.patches as ppatches
import math
import time
import timeit
import functools
import sys

####################################################################
# Utility functions to draw robot, lidar mesaures and occupation map
#
def drawRobot(ax, x,y,theta, robot_diameter, robot_color):
    rradius = robot_diameter / 2
    circle = plt.Circle((x,y), radius=rradius, fill=False, color=robot_color)
    ax.add_artist(circle)
    bearing = plt.Line2D([x, x + rradius * np.cos(theta)], [y, y + rradius * np.sin(theta)], linewidth=2, color=robot_color)
    ax.add_line(bearing)
    ax.axis('equal')
    ax.autoscale()
    return (circle, bearing)

def drawRobotPos(ax, pos, robot_diameter, robot_color):
    x = pos[0]
    y = pos[1]
    theta = pos[2]
    rradius = robot_diameter / 2
    circle = plt.Circle((x,y), radius=rradius, fill=False, color=robot_color)
    ax.add_artist(circle)
    bearing = plt.Line2D([x, x + rradius * np.cos(theta)], [y, y + rradius * np.sin(theta)], linewidth=2, color=robot_color)
    ax.add_line(bearing)
    ax.axis('equal')
    ax.autoscale()
    return (circle, bearing)

def drawOccMapBorder(ax, occ_map_width, occ_map_height):
    rect = ppatches.Rectangle((-occ_map_width / 2.0, -occ_map_height / 2.0), occ_map_width, occ_map_height, edgecolor = 'blue', facecolor = "none")
    ax.add_patch(rect)

def drawOccMap(ax, occ_map, occ_map_width, occ_map_height):
    x_step = occ_map_width / occ_map.shape[0]
    y_step = occ_map_height / occ_map.shape[1]

    for r in range (0, occ_map.shape[0], 1):
        for c in range (0, occ_map.shape[1], 1):
            if occ_map[r, c] < 0.5 :
                rect = ppatches.Rectangle((-occ_map_width / 2.0 + c * x_step, occ_map_height / 2.0 - (r + 1) * y_step), x_step, y_step, edgecolor = 'None', facecolor = "yellow")
                ax.add_patch(rect)


def drawLidarMeasure(ax, pos, measures_points, measures_values):
    x = pos[0]
    y = pos[1]
    theta = pos[2]
    dtheta = 2 * math.pi / measures_points
    lidar_angle = 0
    points_x = np.zeros(measures_points)
    points_y = np.zeros(measures_points)

    for i in range(0, measures_points):
        dx = measures_values[i] * math.cos(theta - math.pi + lidar_angle)
        dy = measures_values[i] * math.sin(theta - math.pi + lidar_angle)
        points_x[i] = x + dx
        points_y[i] = y + dy
        lidar_angle += dtheta

    lidarPoints = ax.scatter(points_x, points_y, c = 'r',marker = '.')

    return lidarPoints

def updatedLidarDraw(lidarPoints, pos, new_data_values, lidar_resolution, map_center, map_scale):
    x = pos[0] * map_scale[0] + map_center[0]
    y = pos[1] * map_scale[1] + map_center[1]
    theta = pos[2]
    dtheta = 2 * math.pi / lidar_resolution
    lidar_angle = 0
    pts_x = np.zeros(lidar_resolution)
    pts_y = np.zeros(lidar_resolution)
    for i in range(0, lidar_resolution):
        dx = new_data_values[i] * math.cos(theta - math.pi + lidar_angle)
        dy = new_data_values[i] * math.sin(theta - math.pi + lidar_angle)
        pts_x[i] = x + dx * map_scale[0]
        pts_y[i] = y + dy * map_scale[1]
        lidar_angle += dtheta

    lidarPoints.set_offsets(np.c_[pts_x, pts_y])
