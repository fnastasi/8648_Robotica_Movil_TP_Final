#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 01:08:29 2021

@author: federico
"""

from math import *
#import math
import numpy as np
import numpy.linalg as lng
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#from medicion_esperada import medicion_esperada

class EKF():
    def __init__(self, mu_0, sigma_0, alfa, T, l, p):
        self.mu = mu_0
        self.sigma = sigma_0
        
        self.noise_param = alfa
        self.dt = T
        
        self.lidar_resolution = l
        self.lidar_pos = p
        
        #self.occ_map = occupation_map
        #self.occ_map_scale = occupation_map_scale
        self.occupation_map_loaded = False
        
        
    def loadoccmap(self,occupation_map,occupation_map_scale):
        self.occ_map = occupation_map
        self.occ_map_scale = occupation_map_scale
        self.occ_map_loaded = True
        self.occ_map_center = np.array(self.occ_map.shape) / 2.0
        
    def get_mean_position(self):
        return self.mu

    def get_covariance(self):
        return self.sigma

    def motion_model_velocity(self,od_new,od_old):
        
        den = (od_old[1] - od_new[1]) * np.cos(od_old[2]) - (od_old[0] - od_new[0]) * np.sin(od_old[2])
        
        if(den != 0):
            
            num = (od_old[0] - od_new[0]) * np.cos(od_old[2]) + (od_old[1] - od_new[1]) * np.sin(od_old[2])
    
            u = 0.5 * (num/den)
            print(u)
        
            # if(u > 999):
            #     u = 999
    
            x_estrella = (od_old[0] + od_new[0]) / 2 + u * (od_old[1] - od_new[1])
            y_estrella = (od_old[1] + od_new[1]) / 2 + u * (od_new[0] - od_old[0])
            r_estrella = np.sqrt((od_old[0] - x_estrella)**2 + (od_old[1] - y_estrella)**2)
    
            delta_theta = np.arctan2(od_new[1] - y_estrella,od_new[0] - x_estrella) - np.arctan2(od_old[1] - y_estrella,od_old[0] - x_estrella)
    
            v = delta_theta * r_estrella/ self.dt
            w = delta_theta / self.dt
            #gamma = (od_new[2] - od_old[2])/self.dt - w


        else:
            v = np.sqrt((od_new[0] - od_old[0])**2 + (od_new[1] - od_old[1])**2)/self.dt
            w = 0
            #gamma = 0
            
        gamma = (od_new[2] - od_old[2])/self.dt - w
            
        return (v,w,gamma)        
        
        
# =============================================================================
# Paso de predicción
# =============================================================================
    def prediction_step(self,od_new,od_old,t):
        
        # od_new: lectura de odometría actual
        # od_old: lectura de odometría anterior
        # utilizando ambasy con el modelo de odometría, se calcula la velocidad
        # del robot, es decir, la velocidad según la odometría
    
        # Cargo la media de la pose del robot, según kalman hasta el momento
        x_avg = self.mu[0]
        y_avg = self.mu[1]
        theta_avg = self.mu[2]
        
        # le digo al filtro de kalman que el paso de tiempo es el que le pasé
        # a la función prediction step
        self.dt = t
        
        # calculo la velocidad y omega en función de las lecturas de odometría.
        # acá no estoy seguro si qué cálculo hacer. Con la función
        # motion_model_velocity() la estimación termina divirgiendo. Por eso
        # uso una forma más sencilla
        
        #(v,w,gamma) = self.motion_model_velocity(od_new,od_old)
        
        v = np.sqrt((od_old[0] - od_new[0])**2 + (od_old[1] - od_new[1])**2)/t
        
        # while(theta_old < 0):
        #     theta_old += 2 * np.pi
            
        # while(theta_new < 0):
        #     theta_new += 2 * np.pi
        
        delta_theta = od_new[2] - od_old[2]
        # while(delta_theta > np.pi):
        #     delta_theta = delta_theta - 2*np.pi
        # while(delta_theta < -np.pi):
        #     delta_theta = delta_theta + 2 * np.pi
        
        w = delta_theta/t
            
        
        #w = (od_new[2] - od_old[2])/t
        #print(w,v)
        # # Dependiendo de si w=0 o no, cambia la expresión de los jacobianos y
        # # de la posición esperada
        # if(w!=0):
        #     v_w = v/w
            
        #     # calculo la nueva pose esperada
        #     x_avg_new = x_avg - v_w * sin(theta_avg) + v_w * cos(theta_avg + w * t)
        #     y_avg_new = y_avg + v_w * cos(theta_avg) - v_w * cos(theta_avg + w * t)
        #     theta_avg_new = theta_avg + w * t
            
        #     v11 = (-sin(theta_avg) + sin(theta_avg + w * t))/w
        #     #v11 = (x_avg_new - x_avg)/v
        #     v12 = (v_w / w) * sin(theta_avg) + (v_w / w) * (cos(theta_avg + w * t) * t * w - sin(theta_avg + w * t))
        #     v21 = (cos(theta_avg) - cos(theta_avg + w * t))/w
        #     #v21 = (y_avg_new - y_avg)/v
        #     v22 = -(v_w / w) * cos(theta_avg) + (v_w/w) * (sin(theta_avg + w * t) * w * t + cos(theta_avg + w * t))
            
            # Jacobiano con respecto al control
            # V = [[v11,v12,0],
            #       [v21,v22,0],
            #       [0,t,t]]
            
            # V = [[v11,v12],
            #       [v21,v22],
            #       [0,t]]
            


            
        
        #else:
            # Usando las velocidades que obtuve a partir de las lecturas de odometría
            # calculo la predicción de la próxima pose del robot

        # theta_avg_new = theta_avg + (od_new[2] - od_old[2])
        # x_avg_new = x_avg + (od_new[0] - od_old[0])
        # y_avg_new = y_avg + (od_new[1] - od_old[1])
        theta_avg_new = theta_avg + w * t
        x_avg_new = x_avg + v * cos(theta_avg_new) * t
        y_avg_new = y_avg + v * sin(theta_avg_new) * t

        
        # V = [[cos(od_old[2]) * t,0],
        #       [sin(od_old[2]) * t,0],
        #       [0,0]]
        
            # Jacobiano del modelo de movimiento respecto de la velocidad lineal
            # y angular. Se usa para calcular la matriz de covarianza de la pose,
            # a partir de la matriz de covarianza de las velocidades.
        # V = [[cos(theta_avg)*t,0],
        #      [sin(theta_avg)*t,0],
        #      [0,t]]
        
        V = [[cos(theta_avg_new)*t, 0],
             [sin(theta_avg_new)*t, 0],
             [0,t]]
        
        # G = [[1, 0, g13],
        #      [0, 1, g23],
        #      [0, 0, 1]] 
        
        G = [[1, 0, y_avg - y_avg_new],
              [0, 1, -x_avg + x_avg_new],
              [0, 0, 1]]
        
        # Jacobiano del modelo de movimiento respecto de la pose anterior
        # G = [[1, 0, -(od_new[1] - od_old[1])],
        #       [0, 1, (od_new[0] - od_old[0])],
        #       [0, 0, 1]]
        # G = [[1, 0, 0],
        #       [0, 1, 0],
        #       [0, 0, 1]]
    
        
        # x_avg = od[0]
        # y_avg = od[1]
        # theta_avg = od[2]
        
        # Matriz de covarianza del modelo de movimiento de velocidad
        
        # Q_aux = [[(self.noise_param[0] * abs(v) + self.noise_param[1] * abs(w))**2 , 0,0],
        #           [0 , (self.noise_param[2] * abs(v) + self.noise_param[3] * abs(w))**2 ,0],
        #           [0,0,(self.noise_param[4] * abs(v) + self.noise_param[5] * abs(w))**2]]
        
        
        Q_aux = [[(self.noise_param[0] * abs(v) + self.noise_param[1] * abs(w))**2,0],
                  [0 , (self.noise_param[2] * abs(v) + self.noise_param[3] * abs(w))**2 ]]
        
        # Calculo la matriz de de covarianza de la pose respecto del modelo de
        # movimiento
        Q = np.dot(np.dot(V,Q_aux),np.transpose(V))
        
        # Predicción de la nueva pose y de la nueva covariana de la pose
        new_sigma = np.dot(np.dot(G, self.sigma), np.transpose(G)) + Q
        
        new_mu = np.array([x_avg_new, y_avg_new, theta_avg_new]).T
        
        while(new_mu[2] > np.pi):
            new_mu[2] = new_mu[2] - 2*np.pi
        while(new_mu[2] < -np.pi):
            new_mu[2] = new_mu[2] + 2 * np.pi

        # se asignan al filtro de Kalman
        self.mu = new_mu
        self.sigma = new_sigma
        
        return new_mu, new_sigma


# =============================================================================
# 
# =============================================================================
    def __calculateCellForPosition(self, x, y):
        norm_pos = np.array([x,y]) / self.occ_map_scale
        pos_occmap = norm_pos + self.occ_map_center

        if pos_occmap[1] > 0:
            row = int(self.occ_map.shape[0] - pos_occmap[1])
        else:
            row = self.occ_map.shape[0] + 1

        if pos_occmap[0] > 0:
            col = int(pos_occmap[0])
        else:
            col = -1
        return (row, col)

# =============================================================================
# 
# =============================================================================
    def __amIOutsideOccMap (self, x, y):
        out_of_map = True
        row = -1
        col = -1
        if self.occ_map_loaded == True:
            (row, col) = self.__calculateCellForPosition(x, y)
            
            if row < 0 or row >= self.occ_map.shape[0]:
                out_of_map = True
            elif col < 0 or col >= self.occ_map.shape[1]:
                out_of_map = True
            else:
                out_of_map = False 
        else:
            out_of_map = True
        
        return (out_of_map, row, col)



# =============================================================================
# 
# =============================================================================
    def __isItFreePosition(self, x, y):
        free_pos = False
        out_of_map = False
        if self.occ_map_loaded == True:
            (out_of_map, row, col) = self.__amIOutsideOccMap(x,y)
            if  out_of_map == False:
                occ_map_value = self.occ_map[row, col]
                if (occ_map_value < 0.5):
                    free_pos = False
                else:
                    free_pos = True
            else:
                free_pos = False
        else:
            free_pos = True

        return (free_pos, out_of_map)


# =============================================================================
# Ray casting
# =============================================================================
    def ray_casting(self, x, y, theta):
            # dx = math.cos(theta) *  self.occ_map_scale
            # dy = math.sin(theta) *  self.occ_map_scale
            dx = np.cos(theta) *  self.occ_map_scale
            dy = np.sin(theta) *  self.occ_map_scale
            free_pos = True
            out_of_map = False
    
            next_x = x
            next_y = y
            last_free_pos_x = x
            last_free_pos_y = y
            while free_pos == True and out_of_map == False:
                last_free_pos_x = next_x
                last_free_pos_y = next_y
                next_x += dx
                next_y += dy
                (free_pos, out_of_map) = self.__isItFreePosition(next_x, next_y)
                if free_pos == False or out_of_map == True:
                    next_x = last_free_pos_x
                    next_y = last_free_pos_y
                    free_pos = True
                    out_of_map = False
                    ddx = dx / 8
                    ddy = dy / 8
                    while free_pos == True and out_of_map == False:
                        last_free_pos_x = next_x
                        last_free_pos_y = next_y
                        next_x += ddx
                        next_y += ddy
                        (free_pos, out_of_map) = self.__isItFreePosition(next_x, next_y)
                else:
                    pass
            
            distance_measured = np.sqrt((last_free_pos_x - x)**2 + (last_free_pos_y - y)**2)
            #error = np.random.normal(scale = 0.015 * distance_measured)
            # error = np.random.normal(scale = 0.15 * distance_measured)
            return  distance_measured


# =============================================================================
# Función que calcula la medición esperada usando raycasting
# =============================================================================
    def medicion_esperada(self):
        # dtheta = 2 * math.pi / self.lidar_resolution
        dtheta = 2 * np.pi / self.lidar_resolution
        lidar_angle = 0
        measure_values = np.zeros(self.lidar_resolution)
        for i in range(0, self.lidar_resolution):
            measure_values[i] = self.ray_casting(self.mu[0], self.mu[1], self.mu[2] - self.lidar_pos[2] + lidar_angle)
            lidar_angle += dtheta
    
        return measure_values
# =============================================================================
# Función de matcheo de puntos: ICP
# =============================================================================
    def closest_point_matching(self,X, P):
      """Performs closest point matching of two point sets.
      
      Arguments:
      X -- reference point set
      P -- point set to be matched with the reference
      
      Output:
      P_matched -- reordered P, so that the elements in P match the elements in X
      """
      
      P_matched = P.copy()
      
      # cantidad de puntos
      N = X.shape[1]
      orden = np.random.permutation(N)
      for l in range(0,N):
          # elijo un punto de la nube de puntos
          i = orden[l]
          x_i = X[:,i]
          # busco el punto más cercano
          dist_min = np.linalg.norm(x_i - P_matched[:,i])
          k = i
          for m in range(l+1,N):
              j = orden[m]
              dist = np.linalg.norm(x_i - P_matched[:,j])
              if(dist<dist_min):
                  x_aux = X[:,j]
                  p_aux = P_matched[:,i]
                  delta_E = dist - dist_min - np.linalg.norm(x_aux - P_matched[:,j]) + np.linalg.norm(x_aux - p_aux)
                  if(delta_E < 0):
                      dist_min = dist
                      k = j
          aux = P_matched[:,k].copy()
          P_matched[:,k] = P_matched[:,i].copy()
          P_matched[:,i] = aux
              
      return P_matched

    def crear_nube_puntos(self,vector):
        
        N = len(vector)
        nube = np.zeros((2,N))
        
        dtheta = 2 * np.pi / N
        lidar_angle = 0
        
        for i in range(N):
            x_i = vector[i] * np.cos(lidar_angle - self.lidar_pos[2])
            y_i = vector[i] * np.sin(lidar_angle - self.lidar_pos[2])
            nube[0,i] = x_i
            nube[1,i] = y_i
            lidar_angle += dtheta
        
        # N = len(vector)
        
        # dtheta = 2 * np.pi / N
        # lidar_angle = 0
        
        # for i in range(N):
        #     if(vector[i] != 0):
        #         x_i = vector[i] * np.cos(lidar_angle - self.lidar_pos[2])
        #         y_i = vector[i] * np.sin(lidar_angle - self.lidar_pos[2])
        #         nube = np.array([[x_i],[y_i]])
        #         break
        #     else:
        #         lidar_angle += dtheta
        
        # for j in range(i,N):
        #     if(vector[j] != 0):
        #         x_i = vector[j] * np.cos(lidar_angle - self.lidar_pos[2])
        #         y_i = vector[j] * np.sin(lidar_angle - self.lidar_pos[2])
        #         aux = np.array([[x_i],[y_i]])
        #         nube = np.hstack((nube,aux))
        #         lidar_angle += dtheta
        
        
        
        return nube


# =============================================================================
# Sub muestreo
# =============================================================================
    def sub_muestrar(self,nube,N):
        
        # vector de muestreo
        nube_sub = nube
        
    
        return nube_sub
        
# =============================================================================
# Matcheo de las mediciones
# =============================================================================

    def matchear_mediciones(self,nube_h_original,nube_measurements,N):

        # creo las nubes de puntos sub muestreadas
        #nube_measurements_sub = self.sub_muestrar(nube_measurements,N)
        
        
        e_old = 1000
        for i in range(10):
            
            #calculate RMSE
            e = 0
            for j in range(0,nube_h_original.shape[1]):
                
              e = e+(nube_h_original[0,j]-nube_measurements[0,j])**2 + (nube_h_original[1,j]-nube_measurements[1,j])**2
            
            e = np.sqrt(e/nube_h_original.shape[1])
            #print("error icp: ", e)
            if(abs(e - e_old) < 1e-3):
                break
            
            e_old = e
            
            #nube_h_sub = self.sub_muestrar(nube_h_original,N)
            
            nube_h_original = self.closest_point_matching(nube_measurements,nube_h_original)
            #nube_h_sub = self.closest_point_matching(nube_measurements_sub,nube_h_sub)
            
            #substract center of mass
            mx = np.transpose([np.mean(nube_measurements,1)])
            mp = np.transpose([np.mean(nube_h_original,1)])
            X_prime = nube_measurements-mx
            P_prime = nube_h_original-mp
            
            # mx = np.transpose([np.mean(nube_measurements_sub,1)])
            # mp = np.transpose([np.mean(nube_h_sub,1)])
            # X_prime = nube_measurements_sub-mx
            # P_prime = nube_h_sub-mp
            
            #singular value decomposition
            W = np.dot(X_prime,np.transpose(P_prime))
            U, s, V = np.linalg.svd(W)
    
            #calculate rotation and translation
            R = np.dot(U,np.transpose(V))
            t = mx-np.dot(R,mp)
        
            #apply transformation
            nube_h_original = np.dot(R,nube_h_original)+t
            
        return nube_h_original
    
# =============================================================================
# decartar_mediciones_invalidas
# =============================================================================

    def descartar_mediciones_invalidas(self,vector):
        N = len(vector)
        
        vector_nuevo = np.zeros(vector.shape) 
        
        for i in range(N):
            if(vector[i] != 0):
                vector_nuevo = np.copy(vector[i])
                break
        if(i<N):
            
            for j in range(i+1,N):
                if(vector[j] != 0):
                    vector_nuevo = np.hstack((vector_nuevo,vector[j]))
        
        return vector_nuevo
                
        
# =============================================================================
# Paso de corrección
# =============================================================================
    def correction_step(self,measurements,t):
        
        #print("correction step")
        
        # Cargo la media de la pose del robot, según kalman hasta el momento
        x_avg = self.mu[0]
        y_avg = self.mu[1]
        theta_avg = self.mu[2]
        
        # le digo al filtro de kalman que el paso de tiempo es el que le pasé
        # a la función correction step
        self.dt = t
        
        # # cantidad de mediciones
        # N = len(measurements)
        
        # # inicializo el jacobiano del modelo de medición
        # H = np.zeros((N,3))
        
        # medición esperada para el paso de corrección. Se calcula usando 
        # ray casting tomando como posición la media de la pose actual. Esto 
        # tarda un poco, pero no encontré otra forma de hacerlo
        h_original = self.medicion_esperada()
        
        # descarto las medicioens inválidas
        h_original = self.descartar_mediciones_invalidas(h_original)
        measurements = self.descartar_mediciones_invalidas(measurements)
        
        
        # ajusto las dimensiones de ambos
        N_1 = len(h_original)
        N_2 = len(measurements)
        M = min(N_1,N_2)
        h_original = h_original[0:M]
        measurements = measurements[0:M]
        
        # inicializo el jacobiano del modelo de medición
        H = np.zeros((M,3))
        
        # creo las nubes de puntos a partir de las mediciones
        nube_h_original = self.crear_nube_puntos(h_original)
        nube_measurements = self.crear_nube_puntos(measurements)
        
        # a partir de ambas nubes de puntos, hago el matcheo de ambas
        nube_h_original = self.matchear_mediciones(nube_h_original,nube_measurements,M)


        # a partir de la nube de puntos, calculo las distancias
        h = np.zeros(M)
        for i in range(M):
            h[i] = np.linalg.norm(nube_h_original[:,i])
        
        #diff = 0
        
        # Se calcula el Jacobiano evaluado en la media de la pose
        dtheta = 2 * np.pi / M
        lidar_angle = 0
        
        key = 1
        
        #R_aux = np.eye(N)
        R_aux = np.eye(M)
        #for i in range(N):
        for i in range(M):
            #if((measurements[i] !=0)):
                 
                alfa = theta_avg - self.lidar_pos[2] + lidar_angle
                H1 = -cos(alfa)
                H2 = -sin(alfa)
                H3 = 0
                
                aux = np.array([H1,H2,H3])
                
                if(key == 1):
                    #print("hola")
                    H = aux
                    diff = np.array([measurements[i] - h[i]])
                    R_aux = np.array([(0.15 * measurements[i])**2])
                    #print(R_aux)
                    key = 0
                    
                else:
                    
                    #H[i,:]= [H1,H2,H3]
                    H = np.vstack((H,aux))
                
                    diff = np.hstack((diff,(measurements[i] - h[i])))
            
                # matriz de covarianza de medición. Es diagonal

                #R[i,i] = (0.15 * measurements[i])**2
                    R_aux = np.hstack((R_aux,(0.15 * measurements[i])**2))

            #else:
                #print("te aviso")
                lidar_angle += dtheta
        
        R = np.diag(R_aux)
        
        # Constantes del filtro de Kalman
        S = np.dot(np.dot(H,self.sigma),np.transpose(H)) + R
        K = np.dot(np.dot(self.sigma,np.transpose(H)),np.linalg.inv(S))
         
        # Se calculan la nueva media y varianza
        mu_new = self.mu + np.dot(K,diff)
        #mu_new = self.mu + np.dot(K,measurements - h)
        sigma_new = np.dot(np.identity(3) - np.dot(K,H),self.sigma)
        
        # normalizo el ángulo de la pose
        while(mu_new[2] > np.pi):
            mu_new[2] = mu_new[2] - 2 * np.pi
        while(mu_new[2] < -np.pi):
            mu_new[2] = mu_new[2] + 2 * np.pi
        
        self.mu = mu_new
        self.sigma = sigma_new
             
        return mu_new, sigma_new






