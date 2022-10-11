# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        N = len(track_list) # N tracks
        M = len(meas_list) # M measurements
        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))
        
        # initialize association matrix
        self.association_matrix = np.inf*np.ones((N,M)) 

        # loop over all tracks and all measurements to set up association matrix
        for i in range(N): 
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]
                dist = self.MHD(track, meas, KF)
                if self.gating(dist,meas.sensor):
                    self.association_matrix[i,j] = dist
        # # the following only works for at most one track and one measurement
        # self.association_matrix = np.matrix([]) # reset matrix
        # self.unassigned_tracks = [] # reset lists
        # self.unassigned_meas = []
        
        # if len(meas_list) > 0:
        #     self.unassigned_meas = [0]
        # if len(track_list) > 0:
        #     self.unassigned_tracks = [0]
        # if len(meas_list) > 0 and len(track_list) > 0: 
        #     self.association_matrix = np.matrix([[0]])
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############
        min_index = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape)
        tracks_id, meas_id = min_index
        # tracks_id = min_index//self.association_matrix.shape[0]
        # meas_id = min_index%self.association_matrix.shape()[0]
        if self.association_matrix[tracks_id,meas_id] == np.inf:
            return np.nan, np.nan
        else:
            # update association_matrix
            self.association_matrix = np.delete(self.association_matrix,tracks_id,axis=0)
            self.association_matrix = np.delete(self.association_matrix,meas_id,axis=1)
            # update track_list and meas_list
            update_track = self.unassigned_tracks[tracks_id]
            update_meas = self.unassigned_meas[meas_id]
            self.unassigned_tracks.remove(update_track)
            self.unassigned_meas.remove(update_meas)

        # # the following only works for at most one track and one measurement
        # update_track = 0
        # update_meas = 0
        
        # # remove from list
        # self.unassigned_tracks.remove(update_track) 
        # self.unassigned_meas.remove(update_meas)
        # self.association_matrix = np.matrix([])
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        probability_threshold = params.gating_threshold
        DOF = sensor.dim_meas
        # DOF = 3
        if MHD < chi2.ppf(probability_threshold,DOF):
            return True
        else:
            return False
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        # calc Mahalanobis distance
        # H = np.matrix([[1, 0, 0, 0, 0, 0],
        #                [0, 1, 0, 0, 0, 0],
        #                [0, 0, 1, 0, 0, 0]]) 
        
        H = meas.sensor.get_H(track.x)
        # gamma = meas.z - H*track.x
        gamma = meas.z - meas.sensor.get_hx(track.x)
        # S = H*track.P*H.transpose() + meas.R
        S = KF.S(track, meas)
        MHD = gamma.transpose()*np.linalg.inv(S)*gamma # Mahalanobis distance formula
        return MHD
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
        print("association matrix :\n",self.association_matrix)
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        # if len(self.unassigned_tracks) > 0:
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        # for track in manager.track_list:            
        #     print('track', track.id, 'score =', track.score)