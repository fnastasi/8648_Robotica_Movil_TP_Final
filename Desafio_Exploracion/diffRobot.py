#module that define a 2D differential robot

#@author: Javier Luiso
#"""

import numpy as np
import matplotlib.image as pimg
# from enum import Enum
import math
import threading
import time
import timeit
import json

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

class LidarTask():
    def __init__(self, robot):
        self.running = True
        self.robot = robot

    def run(self):
        while self.running == True:
            elapsed_time = timeit.timeit(self.robot.doLIDARMeasure, number=1)
            print(elapsed_time)
            time.sleep(0.127)
        print ("finishing LidarTask...")

    def terminate(self): 
        self.running = False


class DiffRobot():
    def __init__(self, x=0, y=0, theta=0):
        #The robot pose comprises its two-dimensional planar coordinates relative to
        #an external coordinate frame, along with its angular orientation.
        #time step for simulation is setted to default value 1 seg.
        self.__ground_truth = (x, y, theta)
        self.__odometry = (x, y, theta)
        self.dt = 1.0

        #Kinematics parameters 
        #values expressed in meters
        self.r = 0.072
        self.l = 0.235

        #Robot dimensions
        #Robot shape = cylinder
        #values expressed in meters
        self.height = 0.10
        self.diameter = 0.35

        #Error parameters
        self.alpha = (0.10, 0.20, 0.10, 0.20, 0.25, 0.25)

        #Occupation Map
        self.occupation_map_loaded = False
        self.occupation_map_scale = 0

        #Sensores
        # BUMPER
        # a value == True means next robot ground_truth value has reached an occupied cell (collision)
        self.__collision_detected = False
        # Sensor activate status
        self.__bumper_sensor_enabled = False
        # A value == True means sensor is pressed, the robot has collided against an obstacle
        self.__bumper_sensor_value = False
        # If bumper is enabled and robot has collided, while tetha is quadrant marked as True, the robot only will accept v < 0.
        self.__theta_qudrants_stopped = (False, False, False, False)
        # If robot reach map_border
        self.__map_border = False

        # LIDAR
        # Lidar position (L_x, L_y, L_theta) respect robot ground_truth.
        self.__lidar_pos = (0.09, 0.0, math.pi)
        self.__lidar_dr_center_robot = 0.00
        self.__lidar_resolution = 144
        self.__lidar_measure = np.zeros(self.__lidar_resolution)
        # Lidar thread
        self.__lidar_sensor_enabled = True


        # Simulation Thread
        self.__continue_running = False
        self.__input_v = 0.0
        self.__input_w = 0.0
        self.__simulation_thread = threading.Thread(target=self.__simulationTask) 
        self.__print_status = False
        self.__step_counter = 0
        self.__simulation_paused = False

        # Record Robot Stauts
        self.__record_status_on =  False
        self.__status_history = []
        


    #private methods
    def __sample_nd(self, b):
        return (b / 6.0) * np.sum(2.0 * np.random.rand(12) - 1.0)


    def __raycast(self, x, y, theta):
        dx = math.cos(theta) *  self.occupation_map_scale
        dy = math.sin(theta) *  self.occupation_map_scale
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
        
        distance_measured = math.sqrt((last_free_pos_x - x)**2 + (last_free_pos_y - y)**2)
        error = np.random.normal(scale = 0.015 * distance_measured)
        return  distance_measured + self.__lidar_dr_center_robot + error


    def __doLIDARMeasure(self):
        dtheta = 2 * math.pi / self.__lidar_resolution
        lidar_angle = 0
        measure_values = np.zeros(self.__lidar_resolution)
        for i in range(0, self.__lidar_resolution):
            measure_values[i] = self.__raycast(self.__ground_truth[0], self.__ground_truth[1], self.__ground_truth[2] - self.__lidar_pos[2] + lidar_angle)
            lidar_angle += dtheta

        self.__lidar_measure = measure_values
        return measure_values
        
    def printStatus(self, elapsed_time):
        if self.__collision_detected == True:
            print('{:d} dt={:.{prec}f} u_t=({:.{prec}f} {:.{prec}f}) GT=({:.{prec}f} {:.{prec}f} {:.{prec1}f} deg) OD=({:.{prec}f} {:.{prec}f} {:.{prec1}f} deg) B={:d} MP={:d} COLLISION'.format(
            self.__step_counter, elapsed_time, self.__input_v, self.__input_w,
            self.__ground_truth[0], self.__ground_truth[1], self.__ground_truth[2]*180 / np.pi,
            self.__odometry[0], self.__odometry[1], self.__odometry[2]*180 / np.pi, self.__bumper_sensor_value, self.__map_border, prec=2, prec1=0))
        else:
            print('{:d} dt={:.{prec}f} u_t=({:.{prec}f} {:.{prec}f}) GT=({:.{prec}f} {:.{prec}f} {:.{prec1}f} deg) OD=({:.{prec}f} {:.{prec}f} {:.{prec1}f} deg) B={:d} MP={:d}'.format(
            self.__step_counter, elapsed_time, self.__input_v, self.__input_w,
            self.__ground_truth[0], self.__ground_truth[1], self.__ground_truth[2]*180 / np.pi,
            self.__odometry[0], self.__odometry[1], self.__odometry[2]*180 / np.pi, self.__bumper_sensor_value, self.__map_border, prec=2, prec1=0))

    def __recordStatus(self):
        status_entry = []

        # Entry Type
        status_entry.append('SIM_STEP')
        # Step counter
        status_entry.append(self.__step_counter)
        # dt value
        status_entry.append(self.dt)
        # Input
        status_entry.append([self.__input_v, self.__input_w])
        # Ground truth
        status_entry.append(self.__ground_truth)
        # Odometry
        status_entry.append(self.__odometry)
        # Bumper sensor status
        status_entry.append(self.__bumper_sensor_enabled)
        # Bumper sensor value
        status_entry.append(self.__bumper_sensor_value)
        # Collision detected
        status_entry.append(self.__collision_detected)
        # Map border status
        status_entry.append(self.__map_border)
        # Lidar Sensor status
        status_entry.append(self.__lidar_sensor_enabled)

        if self.__lidar_sensor_enabled == True:
            status_entry.append(self.__lidar_measure.tolist())

        self.__status_history.append(status_entry)

    def __recordOccupationMap(self):
        occmap_record = []
        occmap_record.append('OCC_MAP')
        occmap_record.append(self.occupation_map_scale)
        occmap_record.append(self.occupation_map.tolist())

        self.__status_history.append(occmap_record)

    def __simulationStep(self):
        self.foward(self.__input_v, self.__input_w)
        self.__step_counter += 1
        lidar_measurement_time = 0.0
        if self.__lidar_sensor_enabled == True:
            lidar_measurement_time = timeit.timeit(self.__doLIDARMeasure, number=1)

        if self.__record_status_on == True:
            self.__recordStatus()

        time_to_sleep = self.dt - lidar_measurement_time
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        else:
            time.sleep(0.010)

    def __simulationTask(self):
        while self.__continue_running == True:
            if self.__simulation_paused == False:
                elapsed_time = timeit.timeit(self.__simulationStep, number=1)
                if self.__print_status == True:
                    self.printStatus(elapsed_time)
            else:
                time.sleep(self.dt)
                
        print("finishing simulation thread...")
        return

    # Public Simulation Control Interface
    def startSimulation(self, print_status = False):
        if self.__continue_running == False:
            self.__print_status = print_status
            self.__continue_running = True
            self.__simulation_thread.start()
        self.__step_counter = 0

    def pauseSimulation(self):
        self.__simulation_paused = True

    def continueSimulation(self):
        self.__simulation_paused = False

    def stopSimulation(self):
        self.__continue_running = False
        self.__simulation_thread.join()
        print("Simulation Thread Stopped")

    def getSimStepCounter(self):
        return self.__step_counter

    def getSimulationTime(self):
        return self.__step_counter * self.dt

    # Public Lidar Control Interface
    def turnLidar_ON(self):
        self.__lidar_sensor_enabled = True
 
    def turnLidar_OFF(self):
        self.__lidar_sensor_enabled = True

    def getLidarMeaurement(self):
        return self.__lidar_measure

    def getLidarResolution(self):
        return self.__lidar_resolution

    # Public Record Status Interface
    def startRecordStatus(self):
        if self.__status_history:
            self.__status_history.clear()

        if self.occupation_map_loaded == True:
            self.__recordOccupationMap()

        self.__record_status_on = True

    def stopRecordStatus(self):
        self.__record_status_on = False

    def saveRecordStatus(self, filename):
        if self.__record_status_on == False:
            with open(filename, 'w') as outfile:
                json.dump(self.__status_history, outfile, indent=1)

    #public methods
    def setErrorParams(self, a1, a2, a3, a4, a5, a6):
        self.alpha = (a1, a2, a3, a4, a5, a6)

    def setTimeStep(self, t):
        self.dt = t

    def setLinearVelocity(self, v):
        self.__input_v = v

    def getLinearVelocity(self):
        return self.__input_v

    def getAngularVelocity(self):
        return self.__input_w

    def setAngularVelocity(self, w):
        self.__input_w = w

    def enableBumper(self):
        self.__bumper_sensor_enabled = True

    def disableBumper(self):
        self.__bumper_sensor_enabled = False

    def isBumperEnable(self):
        return self.__bumper_sensor_enabled

    def getBumperSensorValue(self):
        return self.__bumper_sensor_value

    def getTimeStep(self):
        return self.dt

    def getGroundTruth(self):
        return np.array(self.__ground_truth)

    def getOdometry(self):
        return np.array(self.__odometry)

    def foward(self, v, w):
        if self.__bumper_sensor_enabled == True:
            self.__foward_with_bummper(v, w)
        else:
            self.__foward_without_bummper(v, w)

        self.__normalizeThetaValues()
       
    # Occupation Map Interface   
    def loadOccupationMap(self, filename, scale):
        self.occupation_map = pimg.imread(filename)
        self.occupation_map_loaded = True
        self.occupation_map_scale = scale
        self.occupation_map_center = np.array(self.occupation_map.shape) / 2.0

    def getOccMapSizeInWorld(self):
        occ_map_width = self.occupation_map.shape[0] * self.occupation_map_scale
        occ_map_height = self.occupation_map.shape[1] * self.occupation_map_scale
        return (occ_map_width, occ_map_height)

    def getOccMap(self):
        return self.occupation_map

    # Simulation utility functions
    def __normalizeThetaValues(self):
        gt_theta = self.__ground_truth[2]
        od_theta = self.__odometry[2]
        while gt_theta > math.pi:
            gt_theta -= 2 * math.pi
        
        while gt_theta < -math.pi:
            gt_theta += 2 * math.pi

        while od_theta > math.pi:
            od_theta -= 2 * math.pi
        
        while od_theta < -math.pi:
            od_theta += 2 * math.pi

        self.__ground_truth = (self.__ground_truth[0], self.__ground_truth[1], gt_theta)
        self.__odometry = (self.__odometry[0], self.__odometry[1], od_theta)
              
    def __foward_without_bummper(self, v, w):
        #compute odometry
        if abs(w) > 0:
            v_w = v / w
            next_odom_x = self.__odometry[0] - v_w * np.sin(self.__odometry[2]) + v_w * np.sin(self.__odometry[2] + w * self.dt)
            next_odom_y = self.__odometry[1] + v_w * np.cos(self.__odometry[2]) - v_w * np.cos(self.__odometry[2] + w * self.dt)
            next_odom_theta = self.__odometry[2] + w * self.dt
        else:
            next_odom_x = self.__odometry[0]
            next_odom_y = self.__odometry[1]
            next_odom_theta = self.__odometry[2]

        #compute next ground truth
        v_hat = v + self.__sample_nd(self.alpha[0] * np.abs(v) + self.alpha[1] * np.abs(w))
        w_hat = w + self.__sample_nd(self.alpha[2] * np.abs(v) + self.alpha[3] * np.abs(w))
        gamma_hat = self.__sample_nd(self.alpha[4] * np.abs(v) + self.alpha[5] * np.abs(w))

        if abs(w_hat) > 0:
            v_w = (v_hat / w_hat)
            next_gt_x = self.__ground_truth[0] - v_w * np.sin(self.__ground_truth[2]) + v_w * np.sin(self.__ground_truth[2] + w * self.dt)
            next_gt_y = self.__ground_truth[1] + v_w * np.cos(self.__ground_truth[2]) - v_w * np.cos(self.__ground_truth[2] + w * self.dt)
            next_gt_theta = self.__ground_truth[2] + w_hat * self.dt + gamma_hat * self.dt
        else:
            next_gt_x = self.__ground_truth[0]
            next_gt_y = self.__ground_truth[1]
            next_gt_theta = self.__ground_truth[2]

        iteration_counter = 8
        self.__collision_detected = False
        self.__theta_qudrants_stopped = (False, False, False, False)
        self.__ground_truth = self.__checkForCollision(next_gt_x, next_gt_y, next_gt_theta, v_hat, w_hat, gamma_hat, self.dt, iteration_counter)

        if self.__collision_detected == True:
            self.__ground_truth = (self.__ground_truth[0], self.__ground_truth[1], next_gt_theta)

        self.__odometry = (next_odom_x, next_odom_y, next_odom_theta)
        

    def __foward_with_bummper(self, v, w):
        current_gt = self.__ground_truth
        
        if self.__bumper_sensor_value == False:
            #compute odometry
            if abs(w) > 0:
                v_w = v / w
                next_odom_x = self.__odometry[0] - v_w * np.sin(self.__odometry[2]) + v_w * np.sin(self.__odometry[2] + w * self.dt)
                next_odom_y = self.__odometry[1] + v_w * np.cos(self.__odometry[2]) - v_w * np.cos(self.__odometry[2] + w * self.dt)
                next_odom_theta = self.__odometry[2] + w * self.dt
            else:
                next_odom_x = self.__odometry[0]
                next_odom_y = self.__odometry[1]
                next_odom_theta = self.__odometry[2]

            #compute next ground truth
            v_hat = v + self.__sample_nd(self.alpha[0] * np.abs(v) + self.alpha[1] * np.abs(w))
            w_hat = w + self.__sample_nd(self.alpha[2] * np.abs(v) + self.alpha[3] * np.abs(w))
            gamma_hat = self.__sample_nd(self.alpha[4] * np.abs(v) + self.alpha[5] * np.abs(w))

            if abs(w_hat) > 0:
                v_w = (v_hat / w_hat)
                next_gt_x = self.__ground_truth[0] - v_w * np.sin(self.__ground_truth[2]) + v_w * np.sin(self.__ground_truth[2] + w * self.dt)
                next_gt_y = self.__ground_truth[1] + v_w * np.cos(self.__ground_truth[2]) - v_w * np.cos(self.__ground_truth[2] + w * self.dt)
                next_gt_theta = self.__ground_truth[2] + w_hat * self.dt + gamma_hat * self.dt
            else:
                next_gt_x = self.__ground_truth[0]
                next_gt_y = self.__ground_truth[1]
                next_gt_theta = self.__ground_truth[2]

            iteration_counter = 8
            self.__collision_detected = False
            self.__theta_qudrants_stopped = (False, False, False, False)
            self.__ground_truth = self.__checkForCollision(next_gt_x, next_gt_y, next_gt_theta, v_hat, w_hat, gamma_hat, self.dt, iteration_counter)
            
            if self.__collision_detected == True:
                self.__bumper_sensor_value = True

            self.__odometry = (next_odom_x, next_odom_y, next_odom_theta)

        else:
            # chequear theta para ver si adminitmos valores positivo de v
            can_move_forward = False

            if self.__gt_theta_in_1st_cuadrant() == True:
                if self.__theta_qudrants_stopped[0] == True:
                    can_move_forward = False
                else:
                    can_move_forward = True

            elif self.__gt_theta_in_2nd_cuadrant() == True:
                if self.__theta_qudrants_stopped[1] == True:
                    can_move_forward = False
                else:
                    can_move_forward = True

            elif self.__gt_theta_in_3th_cuadrant() == True:
                if self.__theta_qudrants_stopped[2] == True:
                    can_move_forward = False
                else:
                    can_move_forward = True
            elif self.__gt_theta_in_4th_cuadrant() == True:
                if self.__theta_qudrants_stopped[3] == True:
                    can_move_forward = False
                else:
                    can_move_forward = True
            else:
                pass

            if can_move_forward == False:
                if v > 0:
                    next_odom_theta = self.__odometry[2] + w * self.dt
                    w_hat = w + self.__sample_nd(self.alpha[3] * np.abs(w))
                    gamma_hat = self.__sample_nd(self.alpha[5] * np.abs(w))
                    next_gt_theta = self.__ground_truth[2] + w_hat * self.dt + gamma_hat * self.dt

                    self.__ground_truth = (self.__ground_truth[0], self.__ground_truth[1], next_gt_theta)
                    self.__odometry = (self.__odometry[0], self.__odometry[1], next_odom_theta)
                else:
                    if abs(w) > 0:
                        v_w = v / w
                        next_odom_x = self.__odometry[0] - v_w * np.sin(self.__odometry[2]) + v_w * np.sin(self.__odometry[2] + w * self.dt)
                        next_odom_y = self.__odometry[1] + v_w * np.cos(self.__odometry[2]) - v_w * np.cos(self.__odometry[2] + w * self.dt)
                        next_odom_theta = self.__odometry[2] + w * self.dt
                    else:
                        next_odom_x = self.__odometry[0]
                        next_odom_y = self.__odometry[1]
                        next_odom_theta = self.__odometry[2]

                    #compute next ground truth
                    v_hat = v + self.__sample_nd(self.alpha[0] * np.abs(v) + self.alpha[1] * np.abs(w))
                    w_hat = w + self.__sample_nd(self.alpha[2] * np.abs(v) + self.alpha[3] * np.abs(w))
                    gamma_hat = self.__sample_nd(self.alpha[4] * np.abs(v) + self.alpha[5] * np.abs(w))

                    if abs(w_hat) > 0:
                        v_w = (v_hat / w_hat)
                        next_gt_x = self.__ground_truth[0] - v_w * np.sin(self.__ground_truth[2]) + v_w * np.sin(self.__ground_truth[2] + w * self.dt)
                        next_gt_y = self.__ground_truth[1] + v_w * np.cos(self.__ground_truth[2]) - v_w * np.cos(self.__ground_truth[2] + w * self.dt)
                        next_gt_theta = self.__ground_truth[2] + w_hat * self.dt + gamma_hat * self.dt
                    else:
                        next_gt_x = self.__ground_truth[0]
                        next_gt_y = self.__ground_truth[1]
                        next_gt_theta = self.__ground_truth[2]

                    iteration_counter = 8
                    self.__collision_detected = False
                    self.__theta_qudrants_stopped = (False, False, False, False)
                    self.__ground_truth = self.__checkForCollision(next_gt_x, next_gt_y, next_gt_theta, v_hat, w_hat, gamma_hat, self.dt, iteration_counter)
                    self.__odometry = (next_odom_x, next_odom_y, next_odom_theta)
                    
                    if self.__collision_detected == True:
                        self.__ground_truth = current_gt
                    else:
                        self.__bumper_sensor_value = False

            else:

                #compute odometry
                if abs(w) > 0:
                    v_w = v / w
                    next_odom_x = self.__odometry[0] - v_w * np.sin(self.__odometry[2]) + v_w * np.sin(self.__odometry[2] + w * self.dt)
                    next_odom_y = self.__odometry[1] + v_w * np.cos(self.__odometry[2]) - v_w * np.cos(self.__odometry[2] + w * self.dt)
                    next_odom_theta = self.__odometry[2] + w * self.dt
                else:
                    next_odom_x = self.__odometry[0] + v * np.sin(self.__odometry[2])
                    next_odom_y = self.__odometry[1] + v * np.cos(self.__odometry[2])
                    next_odom_theta = self.__odometry[2]

                #compute next ground truth
                v_hat = v + self.__sample_nd(self.alpha[0] * np.abs(v) + self.alpha[1] * np.abs(w))
                w_hat = w + self.__sample_nd(self.alpha[2] * np.abs(v) + self.alpha[3] * np.abs(w))
                gamma_hat = self.__sample_nd(self.alpha[4] * np.abs(v) + self.alpha[5] * np.abs(w))

                if abs(w_hat) > 0:
                    v_w = (v_hat / w_hat)
                    next_gt_x = self.__ground_truth[0] - v_w * np.sin(self.__ground_truth[2]) + v_w * np.sin(self.__ground_truth[2] + w * self.dt)
                    next_gt_y = self.__ground_truth[1] + v_w * np.cos(self.__ground_truth[2]) - v_w * np.cos(self.__ground_truth[2] + w * self.dt)
                    next_gt_theta = self.__ground_truth[2] + w_hat * self.dt + gamma_hat * self.dt
                else:
                    next_gt_x = self.__ground_truth[0]
                    next_gt_y = self.__ground_truth[1]
                    next_gt_theta = self.__ground_truth[2]

                iteration_counter = 8
                self.__collision_detected = False
                self.__theta_qudrants_stopped = (False, False, False, False)
                self.__ground_truth = self.__checkForCollision(next_gt_x, next_gt_y, next_gt_theta, v_hat, w_hat, gamma_hat, self.dt, iteration_counter)
                
                if self.__collision_detected == True:
                    self.__bumper_sensor_value = True
                else:
                    self.__bumper_sensor_value = False

                self.__odometry = (next_odom_x, next_odom_y, next_odom_theta)

    def __calculateCellForPosition(self, x, y):
        norm_pos = np.array([x,y]) / self.occupation_map_scale
        pos_occmap = norm_pos + self.occupation_map_center

        if pos_occmap[1] > 0:
            row = int(self.occupation_map.shape[0] - pos_occmap[1])
        else:
            row = self.occupation_map.shape[0] + 1

        if pos_occmap[0] > 0:
            col = int(pos_occmap[0])
        else:
            col = -1
        return (row, col)

    def __gt_theta_in_1st_cuadrant(self):
        return 0 < self.__ground_truth[2] and self.__ground_truth[2] < math.pi/2

    def __gt_theta_in_2nd_cuadrant(self):
        return math.pi/2 < self.__ground_truth[2] and self.__ground_truth[2] < math.pi

    def __gt_theta_in_3th_cuadrant(self):
        return -math.pi < self.__ground_truth[2] and self.__ground_truth[2] < -math.pi/2

    def __gt_theta_in_4th_cuadrant(self):
        return -math.pi/2 < self.__ground_truth[2] and self.__ground_truth[2] < 0

    def __amIOutsideOccMap (self, x, y):
        out_of_map = True
        row = -1
        col = -1
        if self.occupation_map_loaded == True:
            (row, col) = self.__calculateCellForPosition(x, y)
            
            if row < 0 or row >= self.occupation_map.shape[0]:
                out_of_map = True
            elif col < 0 or col >= self.occupation_map.shape[1]:
                out_of_map = True
            else:
                out_of_map = False 
        else:
            out_of_map = True
        
        return (out_of_map, row, col)

    def __isItFreePosition(self, x, y):
        free_pos = False
        out_of_map = False
        if self.occupation_map_loaded == True:
            (out_of_map, row, col) = self.__amIOutsideOccMap(x,y)
            if  out_of_map == False:
                occ_map_value = self.occupation_map[row, col]
                if (occ_map_value < 0.5):
                    free_pos = False
                else:
                    free_pos = True
            else:
                free_pos = False
        else:
            free_pos = True

        return (free_pos, out_of_map)

    def __IamInFreeCell(self):
        (center_in_free_cell, out_of_map) = self.__isItFreePosition(self.__ground_truth[0], self.__ground_truth[1])
        robot_in_free_cell = False

        if center_in_free_cell == True:
            (top_border_in_free_cell, out_of_map) = self.__isItFreePosition(self.__ground_truth[0], self.__ground_truth[1] + self.diameter / 2.0)
            (bottom_border_in_free_cell, out_of_map) = self.__isItFreePosition(self.__ground_truth[0], self.__ground_truth[1] - self.diameter / 2.0)
            (left_border_in_free_cell, out_of_map) = self.__isItFreePosition(self.__ground_truth[0] - self.diameter / 2.0, self.__ground_truth[1])
            (right_border_in_free_cell, out_of_map) = self.__isItFreePosition(self.__ground_truth[0] + self.diameter / 2.0, self.__ground_truth[1])

            if top_border_in_free_cell and bottom_border_in_free_cell and left_border_in_free_cell and right_border_in_free_cell:
                robot_in_free_cell = True
            else:
                robot_in_free_cell = False

        else:
            robot_in_free_cell = False

        return (robot_in_free_cell, out_of_map)

    def __willBeInAFreeCell(self, next_gt_x, next_gt_y):
        (center_in_free_cell, out_of_map) = self.__isItFreePosition(next_gt_x, next_gt_y)
        robot_in_free_cell = False

        if center_in_free_cell == True:
            (top_border_in_free_cell, out_of_map) = self.__isItFreePosition(next_gt_x, next_gt_y + self.diameter / 2.0)
            (bottom_border_in_free_cell, out_of_map) = self.__isItFreePosition(next_gt_x, next_gt_y - self.diameter / 2.0)
            (left_border_in_free_cell, out_of_map) = self.__isItFreePosition(next_gt_x - self.diameter / 2.0, next_gt_y)
            (right_border_in_free_cell, out_of_map) = self.__isItFreePosition(next_gt_x + self.diameter / 2.0, next_gt_y)

            if top_border_in_free_cell and bottom_border_in_free_cell and left_border_in_free_cell and right_border_in_free_cell:
                robot_in_free_cell = True
            else:
                robot_in_free_cell = False
                if top_border_in_free_cell == False and (self.__gt_theta_in_1st_cuadrant() or self.__gt_theta_in_2nd_cuadrant()):
                    self.__theta_qudrants_stopped = (True, True, False, False)
                if bottom_border_in_free_cell == False and (self.__gt_theta_in_3th_cuadrant() or self.__gt_theta_in_4th_cuadrant()):
                    self.__theta_qudrants_stopped = (False, False, True, True)
                if left_border_in_free_cell == False and (self.__gt_theta_in_2nd_cuadrant() or self.__gt_theta_in_3th_cuadrant()):
                    self.__theta_qudrants_stopped = (False, True, True, False)
                if right_border_in_free_cell == False and (self.__gt_theta_in_1st_cuadrant() or self.__gt_theta_in_4th_cuadrant()):
                    self.__theta_qudrants_stopped = (True, False, False, True)
        else:
            robot_in_free_cell = False

        return (robot_in_free_cell, out_of_map)

    def __checkForCollision_old(self, next_gt_x, next_gt_y, next_gt_theta, v_hat, w_hat, gamma_hat, dt, iteration_counter):
        (robot_in_free_cell, out_of_map) = self.__willBeInAFreeCell(next_gt_x, next_gt_y)
        iteration_counter = iteration_counter - 1

        self.__map_border = out_of_map

        if abs(w_hat) > 0:
            if robot_in_free_cell == False:
                self.__collision_detected = True
                v_w = (v_hat / w_hat)
                n_gt_x = self.__ground_truth[0] - v_w * np.sin(self.__ground_truth[2]) + v_w * np.sin(self.__ground_truth[2] + w_hat * 0.9 * dt)
                n_gt_y = self.__ground_truth[1] + v_w * np.cos(self.__ground_truth[2]) - v_w * np.cos(self.__ground_truth[2] + w_hat * 0.9 * dt)
                n_gt_theta = self.__ground_truth[2] + w_hat * 0.9 * dt + gamma_hat * 0.9 * dt
                
                if iteration_counter > 0:
                    return self.__checkForCollision(n_gt_x, n_gt_y, next_gt_theta, v_hat, w_hat, gamma_hat, 0.9 * dt, iteration_counter)
                else:
                    self.__ground_truth = (self.__ground_truth[0], self.__ground_truth[1], next_gt_theta)
                    return self.__ground_truth

            else:
                if self.__collision_detected == True and iteration_counter > 0:
                    v_w = (v_hat / w_hat)
                    n_gt_x = self.__ground_truth[0] - v_w * np.sin(self.__ground_truth[2]) + v_w * np.sin(self.__ground_truth[2] + w_hat * 1.5 * dt)
                    n_gt_y = self.__ground_truth[1] + v_w * np.cos(self.__ground_truth[2]) - v_w * np.cos(self.__ground_truth[2] + w_hat * 1.5 * dt)
                    n_gt_theta = self.__ground_truth[2] + w_hat * dt/2 + gamma_hat * dt/2
                    return self.__checkForCollision(n_gt_x, n_gt_y, next_gt_theta, v_hat, w_hat, gamma_hat, 1.5 * dt, iteration_counter)
                else:
                    self.__ground_truth = (next_gt_x, next_gt_y, next_gt_theta)
       
        return self.__ground_truth

    def __checkForCollision(self, next_gt_x, next_gt_y, next_gt_theta, v_hat, w_hat, gamma_hat, dt, iteration_counter):
        (robot_in_free_cell, out_of_map) = self.__willBeInAFreeCell(next_gt_x, next_gt_y)

        iterations = 10
        dt_step = dt / iterations
        next_dt = 0
        last_know_free_gt = self.__ground_truth

        while robot_in_free_cell == False and iterations > 0:
            self.__collision_detected = True

            next_dt += dt_step
            v_w = (v_hat / w_hat)
            n_gt_x = self.__ground_truth[0] - v_w * np.sin(self.__ground_truth[2]) + v_w * np.sin(self.__ground_truth[2] + w_hat * next_dt)
            n_gt_y = self.__ground_truth[1] + v_w * np.cos(self.__ground_truth[2]) - v_w * np.cos(self.__ground_truth[2] + w_hat * next_dt)
            n_gt_theta = self.__ground_truth[2] + w_hat * next_dt + gamma_hat * next_dt
            (robot_in_free_cell, out_of_map) = self.__willBeInAFreeCell(n_gt_x, n_gt_y)
            if robot_in_free_cell == True:
                last_know_free_gt = (n_gt_x, n_gt_y, n_gt_theta)
            iterations -= 1

        if self.__collision_detected == True:
            self.__ground_truth = last_know_free_gt
        else:
            self.__ground_truth = (next_gt_x, next_gt_y, next_gt_theta)

        return self.__ground_truth
    