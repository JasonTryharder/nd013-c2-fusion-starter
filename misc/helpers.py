# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : helper functions for loop_over_dataset.py
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import os
import pickle
import cv2
import glob

## Saves an object to a binary file
def save_object_to_file(object, file_path, base_filename, object_name, frame_id=1):
    object_filename = os.path.join(file_path, os.path.splitext(base_filename)[0]
                                   + "__frame-" + str(frame_id) + "__" + object_name + ".pkl")
    with open(object_filename, 'wb') as f:
        pickle.dump(object, f)

## Loads an object from a binary file
def load_object_from_file(file_path, base_filename, object_name, frame_id=1):
    object_filename = os.path.join(file_path, os.path.splitext(base_filename)[0]
                                   + "__frame-" + str(frame_id) + "__" + object_name + ".pkl")
    with open(object_filename, 'rb') as f:
        object = pickle.load(f)
        return object
    
## Prepares an exec_list with all tasks to be executed
def make_exec_list(exec_detection, exec_tracking, exec_visualization): 
    
    # save all tasks in exec_list
    exec_list = exec_detection + exec_tracking + exec_visualization
    
    # check if we need pcl
    if any(i in exec_list for i in ('validate_object_labels', 'bev_from_pcl')):
        exec_list.append('pcl_from_rangeimage')
    # check if we need image
    if any(i in exec_list for i in ('show_tracks', 'show_labels_in_image', 'show_objects_in_bev_labels_in_camera')):
        exec_list.append('load_image')
    # movie does not work without show_tracks
    if 'make_tracking_movie' in exec_list:  
        exec_list.append('show_tracks')  
    return exec_list

# help function to save image to file also save image to video for report
def save_video(file_name):
    img = cv2.imread(filename=file_name)
    frame_height,frame_width,depth = img.shape
    # resolve output path
    output_path = '_'.join(file_name.split('_')[0:-1]) + '.mp4'
    fps = 5
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (frame_width,frame_height)
    out_video = cv2.VideoWriter(output_path,codec,fps,frame_size)
    # resolve dir for images
    dir = '/'.join(file_name.split('/')[0:-1])
    # resolve name_pattern for search image to compile for videos
    name_pattern = '_'.join(file_name.split('/')[-1].split('_')[0:-1])
    name_pattern = name_pattern+'*.png'
    name_pattern = os.path.join(dir,name_pattern)
    # use file_name to find the related images
    for each in glob.glob(name_pattern):
        img = cv2.imread(filename=each)
        out_video.write(img)
        # print(each)
    out_video.release()
# save_image will return True is also saved to video
def save_image(img,file_name,dir,index):
    file_name = file_name+'_'+str(index)+'.png'
    file_name = os.path.join(dir,file_name)
    cv2.imwrite(file_name,img)
    if index >=20:
        save_video(file_name=file_name)
        return True
    return False

# global variable cnt_frame
cnt_frame = 0
def init_global_var():
    global cnt_frame
    cnt_frame = 0

def read_global_var():
    return cnt_frame

def set_global_var(num):
    global cnt_frame 
    cnt_frame = num