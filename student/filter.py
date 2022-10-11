# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        # self.H = np.matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
        
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = params.dt
        return np.matrix([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        # return 0
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        # Q = np.zeros(shape=(6,6))
        q = params.q
        a = (q*(params.dt)**3)/3
        b = (q*(params.dt)**2)/2
        c = q*params.dt
        Q = np.matrix([[a,0,0,b,0,0],[0,a,0,0,b,0],[0,0,a,0,0,b],[b,0,0,c,0,0],[0,b,0,0,c,0],[0,0,b,0,0,c]])
        return Q
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        x_prime = self.F() * track.x
        P_prime = self.F() * track.P * np.transpose(self.F()) + self.Q()
        track.set_x(x_prime)
        track.set_P(P_prime)
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        gamma = self.gamma(track,meas)
        S = self.S(track,meas)
        K = track.P * np.transpose(meas.sensor.get_H(track.x)) * np.linalg.inv(S)
        x_prime = track.x + K *gamma
        P_prime = (np.identity(6) -K * meas.sensor.get_H(track.x)) * track.P
        track.set_x(x_prime)
        track.set_P(P_prime)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        return meas.z - meas.sensor.get_hx(track.x)
        
        ############
        # END student code
        ############ 

    def S(self, track, meas):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        return meas.sensor.get_H(track.x) * track.P * np.transpose(meas.sensor.get_H(track.x)) + meas.R
        
        ############
        # END student code
        ############ 