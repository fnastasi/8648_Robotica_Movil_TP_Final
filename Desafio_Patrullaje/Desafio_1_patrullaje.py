#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 17:34:49 2021

@author: federico
"""
# robot para la simulación
from diffRobot import DiffRobot
# librerías varias
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
import matplotlib.patches as ppatches
import math
import time
import timeit
import functools
import sys

import importlib

from scipy import signal

# funciones para hacer ploteos
from ploteo_de_la_simulacion import drawRobot
from ploteo_de_la_simulacion import drawRobotPos
from ploteo_de_la_simulacion import drawLidarMeasure
from ploteo_de_la_simulacion import updatedLidarDraw

# librería del filtro de kalman extendido
from EKF import EKF

# ##################################################################################3

# Pose inicial REAL del robot (ground truth)
x0 = 0
y0 = 0
theta0 = 0

# Creo una instancia de DiffRobot
my_robot = DiffRobot(x0,y0,theta0)

# Cargo el mapa de grilla de ocupación
# Como resolción (metros/píxeles) uso el valor 5 metros/126 píxeles

occupation_map_file = "imagen_2021_mapa_tp.png"
occupation_map_scale_factor = 5/126

my_robot.loadOccupationMap(occupation_map_file, occupation_map_scale_factor)

# se guardan estas variables para hacer los gráficos del mapa y de la ubicación del robot
(occ_map_height, occ_map_width) = my_robot.getOccMapSizeInWorld()
occ_map = my_robot.getOccMap()

########################################################################################
# Se crea el gráfico donde se va a plottear la simulación
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.pcolormesh(np.flipud(occ_map))
ax.axis('off')
status_text = ax.text(0, -5, "")

map_center = (occ_map.shape[0] /2.0, occ_map.shape[1] / 2.0)
map_scale = (occ_map.shape[0] / occ_map_width, occ_map.shape[1] / occ_map_height)

plt.ion()
(gt_icon, gt_bearing) = drawRobot(ax, 0 + map_center[0], 0 + map_center[1], 0, my_robot.diameter * map_scale[0], 'g')
(od_icon, od_bearing) = drawRobot(ax, 0 + map_center[0], 0 + map_center[1], 0, my_robot.diameter * map_scale[0], 'grey')

plt.draw()
plt.pause(1)

lidar_values = np.zeros(my_robot.getLidarResolution())
lidarPoints = drawLidarMeasure(ax, [0 ,0 ,0], my_robot.getLidarResolution(), lidar_values)

#############################################################################################

# Se configura al robot y comienza la simulacón
my_robot.setLinearVelocity(0.2)
my_robot.setAngularVelocity(0)
my_robot.enableBumper()
my_robot.turnLidar_ON()


# variable donde guardo las lecturas del kalman filter para comparar con ground
# truth, luego de finalizada la simulación
valores_ekf = [];
valores_reales = [];

my_robot.startSimulation(print_status=False)
#my_robot.foward(0.2,0)
################################################################################################
# cantidad de iteraciones de la simulación
simulation_steps = 6000

# Estado 1: localización, en este estado se quiere ejectuar el algoritmo de
# localización del robot
robot_state = 1

# variable que indica si está activada la rutina de localización o no
localization_on = False

paso = 1

for i in range(simulation_steps):
    if(paso == 1):
        
        if(my_robot.getGroundTruth()[0] > 0.5):
            my_robot.setLinearVelocity(0)
            my_robot.setAngularVelocity(0.5)
    
        if(my_robot.getGroundTruth()[2] > np.pi/2):
            my_robot.setLinearVelocity(0.2)
            my_robot.setAngularVelocity(0)
            paso = 2
            
        
        
    elif(paso == 2):
    
        if(my_robot.getGroundTruth()[1] > 0.8):
            my_robot.setLinearVelocity(0)
            my_robot.setAngularVelocity(0.5)
        
        if(0 > my_robot.getGroundTruth()[2] > -np.pi):
            my_robot.setLinearVelocity(0.2)
            my_robot.setAngularVelocity(0)
            paso = 3
    
        
    
    # Estado 1: se quiere localizar al robot
    if(robot_state == 1):
        # si no se está ejecutando una rutina de localización, pero se quiere
        # comenzar a ejecutar
        
        if(localization_on == False):
            # x_0 = np.random.uniform(0,occ_map_width)
            # y_0 = np.random.uniform(0,occ_map_height)
            # theta_0 = np.random.uniform(-math.pi,math.pi)
            x_0 = -0.5
            y_0 = -0.5
            theta_0 = 0
            
            # pose estimada inicial del filtro de kalman
            pos_0 = np.array([x_0,y_0,theta_0]).T
            # varianza inicial
            var_0 = np.eye(3)
            # parámetros de error que se le pasan al filtro de kalman para que
            # pueda hacer sus cálculos
            error_parameters = np.array([0.10, 0.20, 0.10, 0.20, 0.25, 0.25])
            
            # inicializo el filtro de kalman
            
            filtro_de_kalman = EKF(pos_0,var_0,error_parameters,my_robot.getTimeStep(),my_robot.getLidarResolution(),np.array([0.09, 0.0, math.pi]))
            
            # le digo al filtro de kalman en qué mapa se va a realizar la
            # localización
            filtro_de_kalman.loadoccmap(occ_map,occupation_map_scale_factor)
            
            # indico que se está ejecutando una rutina de localización
            localization_on = True
            # instantes de tiempo de la simulación lectura de odometría del
            # robot. Se usarán para calcular después las predicciones y 
            # correcciones.
            t_old = my_robot.getSimulationTime()
            od_old = my_robot.getOdometry()
            
        # si se está ejecutando una rutina de localización, se continua con esta
        else:
            
            # consulto el tiempo de simulación actual para calcular el paso de
            # tiempo desde la última lectura de odometría
            t_new = my_robot.getSimulationTime()
            T = t_new - t_old
            if(T != 0):
                # consulto la lectura actual de odometría y se lo paso al 
                # filtro de kalman para que prediga la próxima pose
                od_new = my_robot.getOdometry()
                filtro_de_kalman.prediction_step(od_new,od_old,T)
                
                od_old = od_new
                t_old = t_new
                
                # paso de corrección
                t_new = my_robot.getSimulationTime()
                T = t_new - t_old
                filtro_de_kalman.correction_step(my_robot.getLidarMeaurement(),T)
                t_old = t_new
    
    # me guardo la pose según el filtro de kalman y la pose real para comparar
    # al final de la simulación
    valores_ekf.append(filtro_de_kalman.get_mean_position())
    valores_reales.append(my_robot.getGroundTruth())
    
    # The state of the robot is consulted for drawing purpose
    # To update de draw of robot's status could take more time that
    # simulation step (dt = 0.1 seg.)
    # As simulation run in an independent thread the draw will be refreshing 
    # at a lower frequency.
    od = my_robot.getOdometry()
    gt = my_robot.getGroundTruth()

    # # gt_icon.set_center((gt[0] * map_scale[0] + map_center[0], gt[1] * map_scale[1] + map_center[1])) # only for Ptyhon 3
    # gt_icon.center = gt[0] * map_scale[0] + map_center[0], gt[1] * map_scale[1] + map_center[1]
    # #gt_icon.center = gt[0] * map_scale[0], gt[1] * map_scale[1]
    # gt_bearing.set_xdata([gt[0] * map_scale[0] + map_center[0], gt[0] * map_scale[0] + map_center[0] + 0.5 * map_scale[0] * my_robot.diameter * np.cos(gt[2])])
    # #gt_bearing.set_xdata([gt[0] * map_scale[0], gt[0] * map_scale[0] + 0.5 * map_scale[0] * my_robot.diameter * np.cos(gt[2])])
    # gt_bearing.set_ydata([gt[1] * map_scale[1] + map_center[1], gt[1] * map_scale[1] + map_center[1] + 0.5 * map_scale[1] * my_robot.diameter * np.sin(gt[2])])
    # #gt_bearing.set_ydata([gt[1] * map_scale[1], gt[1] * map_scale[1] + 0.5 * map_scale[1] * my_robot.diameter * np.sin(gt[2])])

    gt_icon.center = gt[0] * map_scale[0] + map_center[0], gt[1] * map_scale[1] + map_center[1]
    gt_bearing.set_xdata([gt[0] * map_scale[0] + map_center[0], gt[0] * map_scale[0] + map_center[0] + 0.5 * map_scale[0] * my_robot.diameter * np.cos(gt[2])])
    gt_bearing.set_ydata([gt[1] * map_scale[1] + map_center[1], gt[1] * map_scale[1] + map_center[1] + 0.5 * map_scale[1] * my_robot.diameter * np.sin(gt[2])])
    
    # # od_icon.set_center((od[0]* map_scale[0]  + map_center[0], od[1] * map_scale[1]  + map_center[1])) # only for Ptyhon 3
    # od_icon.center = od[0]* map_scale[0]  + map_center[0], od[1] * map_scale[1]  + map_center[1]
    # #od_icon.center = od[0]* map_scale[0], od[1] * map_scale[1]
    # od_bearing.set_xdata([od[0] * map_scale[0] + map_center[0], od[0] * map_scale[0] + map_center[0] + 0.5 * my_robot.diameter * map_scale[0] * np.cos(od[2])])
    # #od_bearing.set_xdata([od[0] * map_scale[0], od[0] * map_scale[0] + 0.5 * my_robot.diameter * map_scale[0] * np.cos(od[2])])
    # od_bearing.set_ydata([od[1] * map_scale[1] + map_center[1], od[1] * map_scale[1] + map_center[1] + 0.5 * my_robot.diameter * map_scale[1] * np.sin(od[2])])
    # #od_bearing.set_ydata([od[1] * map_scale[1], od[1] * map_scale[1] + 0.5 * my_robot.diameter * map_scale[1] * np.sin(od[2])])

    od_icon.center = od[0]* map_scale[0]  + map_center[0], od[1] * map_scale[1]  + map_center[1]
    od_bearing.set_xdata([od[0] * map_scale[0] + map_center[0], od[0] * map_scale[0] + map_center[0] + 0.5 * my_robot.diameter * map_scale[0] * np.cos(od[2])])
    od_bearing.set_ydata([od[1] * map_scale[1] + map_center[1], od[1] * map_scale[1] + map_center[1] + 0.5 * my_robot.diameter * map_scale[1] * np.sin(od[2])])
    

    lidar_values = my_robot.getLidarMeaurement()
    updatedLidarDraw(lidarPoints, gt, lidar_values, my_robot.getLidarResolution(), map_center, map_scale)

    v = my_robot.getLinearVelocity()
    w = my_robot.getAngularVelocity()
    s_text = "time = " + "{:.2f}".format(my_robot.getSimulationTime()) + "seg   u_t=(" + "{:.2f}".format(v)  + " ; " + "{:.2f}".format(w) + ") Collision = " + str(my_robot.getBumperSensorValue()) 
    status_text.set_text(s_text)

    plt.draw()
    plt.pause(my_robot.dt)

    # As plot function takes time it is not needed to sleep main thread
    # if it is not the case consider to sleep main thread.
    #time.sleep(my_robot.dt)
  
my_robot.stopSimulation()

x_ekf = []
x_real = []

y_ekf = []
y_real = []

theta_ekf = []
theta_real = []

for i in range(0,len(valores_ekf)):
    x_ekf.append(valores_ekf[i][0])
    x_real.append(valores_reales[i][0])
    
    y_ekf.append(valores_ekf[i][1])
    y_real.append(valores_reales[i][1])
    
    theta_ekf.append(valores_ekf[i][2])
    theta_real.append(valores_reales[i][2])
    
tiempo_sim = np.arange(0,simulation_steps,1)
tiempo_sim = tiempo_sim * my_robot.getSimulationTime() / simulation_steps

fig1, ax1 = plt.subplots(1)
ax1.plot(tiempo_sim[0:simulation_steps],x_ekf[0:simulation_steps],'b-',label = 'EKF')
ax1.plot(tiempo_sim[0:simulation_steps],x_real[0:simulation_steps],'r-',label = 'real')
ax1.set_xlabel('tiempo, [s]')
ax1.set_ylabel('x')
ax1.set_title('posición real vs estimación con EKF, en x')
ax1.grid(True)
ax1.legend()

fig2, ax2 = plt.subplots(1)
ax2.plot(tiempo_sim[0:simulation_steps],y_ekf[0:simulation_steps],'b-',label = 'EKF')
ax2.plot(tiempo_sim[0:simulation_steps],y_real[0:simulation_steps],'r-',label = 'real')
ax2.set_xlabel('tiempo, [s]')
ax2.set_ylabel('y')
ax2.set_title('posición real vs estimación con EKF, en y')
ax2.grid(True)
ax2.legend()


fig3, ax3 = plt.subplots(1)
ax3.plot(tiempo_sim[0:simulation_steps],theta_ekf[0:simulation_steps],'b-',label = 'EKF')
ax3.plot(tiempo_sim[0:simulation_steps],theta_real[0:simulation_steps],'r-',label = 'real')
ax3.set_xlabel('tiempo, [s]')
ax3.set_ylabel('theta')
ax3.set_title('posición real vs estimación con EKF, en theta')
ax3.grid(True)
ax3.legend()



