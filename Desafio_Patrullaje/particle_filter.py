#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:34:18 2021

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
# =============================================================================
# Clase partícula
# =============================================================================
class particle():
    def __init__(self):
        self.x = np.random.uniform(-0.1,0.1)
        self.y = np.random.uniform(-0.1,0.1)
        self.theta = np.random.uniform(-np.pi,np.pi)
        
        self.peso = 1.0
        
        # self.noise_param = alfa
        
        # self.lidar_resolution = l
        # self.lidar_pos = p
        
        # self.occupation_map_loaded = False
    # #========================================================================#
    # def loadoccmap(self,occupation_map,occupation_map_scale):
    #     self.occ_map = occupation_map
    #     self.occ_map_scale = occupation_map_scale
    #     self.occ_map_loaded = True
    #     self.occ_map_center = np.array(self.occ_map.shape) / 2.0
    
    #========================================================================#
    # modelo de movimiento de la partícula
    def mov_odom(self,v,w,t):

        theta_new = self.theta + w * t
        x_new = self.x + v * cos(theta_new) * t
        y_new = self.y + v * sin(theta_new) * t
        
        self.x = x_new
        self.y = y_new
        self.theta = theta_new
# =============================================================================
# Clase filtro de partículas
# =============================================================================
class particle_filter():
    def __init__(self,N,alfa,l,p):
        
        # inicialización del filtro de partículas. Se crean "N" partículas en
        # poses aleatorias
        self.noise_param = alfa
        
        self.lidar_resolution = l
        self.lidar_pos = p
        
        self.occupation_map_loaded = False
        
        self.particles = []
        self.pesos = []
        
        for i in range(0,N):
            # creo una partícula
            p = particle()
            #p.loadoccmap(occupation_map,occupation_map_scale)
            self.particles.append(p)
            self.pesos.append(1.0)
        
        self.N = N
    #========================================================================#
    def mean_particle_pos(self):
        x_mean = 0
        y_mean = 0
        theta_mean = 0
        for i in range(self.N):
            x_mean = x_mean + self.particles[i].x
            y_mean = y_mean + self.particles[i].y
            theta_mean = theta_mean + self.particles[i].theta
        x_mean = x_mean / self.N
        y_mean = y_mean / self.N
        theta_mean = theta_mean / self.N
        
        pos_media = np.array([x_mean,y_mean,theta_mean])
        return pos_media
    #========================================================================#
    def std_err_particle_pos(self):
        # obengo la media
        mu = self.mean_particle_pos()
        x_std = 0
        y_std = 0
        for i in range(self.N):
            x_std = x_std + (self.particles[i].x - mu[0])**2
            y_std = y_std + (self.particles[i].y - mu[1])**2
        
        x_std = np.sqrt((1/self.N) * x_std)
        y_std = np.sqrt((1/self.N) * y_std)
        
        desvio = np.array([x_std,y_std])
        return desvio
        
    #========================================================================#
    def loadoccmap(self,occupation_map,occupation_map_scale):
        self.occ_map = occupation_map
        self.occ_map_scale = occupation_map_scale
        self.occ_map_loaded = True
        self.occ_map_center = np.array(self.occ_map.shape) / 2.0     
    #========================================================================#
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
    #========================================================================#
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
    #========================================================================#
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
    #========================================================================#
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
    #========================================================================#
    def medicion_esperada(self,x,y,theta,paso):
        # dtheta = 2 * math.pi / self.lidar_resolution
        dtheta = 2 * np.pi / self.lidar_resolution
        dtheta = paso * dtheta
        lidar_angle = 0
        measure_values = np.zeros(int(self.lidar_resolution / paso))
        for i in range(0, int(self.lidar_resolution / paso)):
            measure_values[i] = self.ray_casting(x, y, theta - self.lidar_pos[2] + lidar_angle)
            lidar_angle += dtheta
        return measure_values
    #========================================================================#
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
        return nube
    #========================================================================#
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
    
    #========================================================================#
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
            if(abs(e - e_old) < 1e-2):
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
    #========================================================================#
    def resample_particles(self,w):
        
        w_max = max(w)
        w = np.exp(w-w_max)
        
        # sumo los pesos
        # eta = np.sum(w)
        
        # # normalizo los pesos
        # w = np.divide(w,eta)
        
        c = []    
        c.append(w[0])
        
        for i in range(1,self.N):
            c.append(c[i-1] + w[i])
        
        step = 1/(self.N)
        
        seed = np.random.uniform(0,step)
        
        i = 0
        u = seed
        p_sampled = []
        # resample the particles based on the seed , step and cacluated pdf
        for h in range(self.N):
#            '''Write the code here'''
            while u > c[i]:
                i = i + 1
            particula_elegida = self.particles[i]
            particula_elegida.peso = 1.0
            p_sampled.append(particula_elegida)
            u = u + 1/self.N
        
        self.particles = p_sampled
    #========================================================================#
    def update_particles_measurements(self,measurements):
        
        
        paso = 2
        
        # vector de pesos weights
        weights = []
        measurements = measurements[0::paso]
        nube_measurements = self.crear_nube_puntos(measurements)
        M = len(measurements)
        for i in range(self.N):
            
            x = (self.particles[i]).x
            y = (self.particles[i]).y
            theta = (self.particles[i]).theta
            
            # busco el vector de medición esperada
            medicion_esperada = self.medicion_esperada(x,y,theta,paso)
            
            # matcheo
            # creo las nubes de puntos a partir de las mediciones
            nube_h_original = self.crear_nube_puntos(medicion_esperada)

            # a partir de ambas nubes de puntos, hago el matcheo de ambas
            nube_h_original = self.matchear_mediciones(nube_h_original,nube_measurements,M)
            
            # a partir de la nube de puntos, calculo las distancias
            h = np.zeros(M)
            for j in range(M):
                h[j] = np.linalg.norm(nube_h_original[:,j])
            
            diff = h - measurements
            
            variance = (0.15 * measurements)**2
            
            #prob = np.prod(np.exp(-(diff)**2 / (2 * variance)) / (np.sqrt(variance * 2 * np.pi)))
            prob = np.sum(-0.5 * np.log(2 * np.pi * variance) + (-(diff)**2) / (2 * variance))
            weights.append(prob)
            (self.particles)[i].peso = prob
            
        print("ESTOY RESAMPLEANDO")    
        # hago remuestreo
        self.resample_particles(weights)
        
    #========================================================================#
    # paso de actualización según el modelo de movimiento. Se actualizan las
    # poses de todas las partículas
    def update_particles_motion(self,od_old,od_new,t):
        # actualizo la pose de todas las partículas
        
        # actualizo según la odometría
        
        # velocidad lineal
        v = np.sqrt((od_old[0] - od_new[0])**2 + (od_old[1] - od_new[1])**2)/t
        
        # velocidad angular
        w = od_new[2] - od_old[2]/t
        
        for i in range(self.N):
            (self.particles[i]).mov_odom(v,w,t)
        
        
        
        
            
            
        
        
        

        
        
        
        
        
        
        
        
        