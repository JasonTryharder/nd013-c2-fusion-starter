# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import zlib
import open3d as o3d
from misc.helpers import save_image, save_video,read_global_var

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")
    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()
    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    # When encountered error : GLFW Error: GLX: Failed to create context: BadValue (integer parameter out of range for operation)
    # sometime is due to nvidia diriver issue, reboot will solve it 
    pcd.points = o3d.utility.Vector3dVector(pcl[:,0:3])
    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    vis.add_geometry(pcd)
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    def key_callback(vis):
        o3d.visualization.VisualizerWithKeyCallback.destroy_window(vis)
    vis.register_key_callback(262,key_callback)
    vis.run()
    vis.destroy_window()
    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    
    # step 2 : extract the range and the intensity channel from the range image
    ri_range = ri[:,:,0]
    ri_intensity = ri[:,:,1]
    # step 3 : set values <0 to zero
    # make range image invalid point (-1) to (0)
    ri[ri<0]=0.0
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    # since the intensity image is skewered toward high intensity images, we need the below scaling method 
    ri_intensity = np.amax(ri_intensity)/2 * ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
    img_intensity = ri_intensity.astype(np.uint8)
    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    # focus on +/- 45° around the image center
    deg45 = int(img_range.shape[1] / 8)
    ri_center = int(img_range.shape[1]/2)
    img_range = img_range[:,ri_center-deg45:ri_center+deg45]
    img_intensity = img_intensity[:,ri_center-deg45:ri_center+deg45]

    img_range_intensity = np.vstack((img_range,img_intensity))
    # img_range_intensity = [] # remove after implementing all steps
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  
    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")
    # print(lidar_pcl.shape)
    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    x_factor = configs.bev_height/(configs.lim_x[1] - configs.lim_x[0])
    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_cpy = np.copy(lidar_pcl)
    # lidar_pcl_cpy[:,2] = np.int_(lidar_pcl_cpy[:,2])
    lidar_pcl_cpy[:,0] = np.int_(lidar_pcl_cpy[:,0] * x_factor)
    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    y_factor = configs.bev_width/(configs.lim_y[1] - configs.lim_y[0])
    test = lidar_pcl_cpy[:,1]
    lidar_pcl_cpy[:,1] = np.int_(lidar_pcl_cpy[:,1] * y_factor + 0.5*(configs.bev_width + 1))
    test = lidar_pcl_cpy[:,1]
    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    # disable once task finished 
    # show_pcl(lidar_pcl_cpy)
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height+1,configs.bev_width+1))
    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    ind = np.lexsort((-1*lidar_pcl_cpy[:,2],lidar_pcl_cpy[:,1],lidar_pcl_cpy[:,0]))
    lidar_pcl_cpy = lidar_pcl_cpy[ind]
    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _,ind,counts = np.unique(lidar_pcl_cpy[:,0:2], axis=0 ,return_index=True, return_counts=True)
    pcl_intensity = lidar_pcl_cpy[ind]
    print(counts)
    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    pcl_intensity[pcl_intensity[:,3]>1,3] = 1
    print(np.amax(pcl_intensity[:,3]))
    print(np.amin(pcl_intensity[:,3]))
    _,ind = np.unique(pcl_intensity[:,0:2], axis=0, return_index=True)
    pcl_intensity = pcl_intensity[ind]
    intensity_map[np.int_(pcl_intensity[:,0]),np.int_(pcl_intensity[:,1])] = pcl_intensity[:,3]/(np.amax(pcl_intensity[:,3])-np.amin(pcl_intensity[:,3]))
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    vis = True
    view_bev_img(intensity_map,vis=False,save_to_video=False,file_name='BEV_intensity_img_step2')
    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height+1,configs.bev_width+1))
    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    ind = np.lexsort((-1*lidar_pcl_cpy[:,2],lidar_pcl_cpy[:,1],lidar_pcl_cpy[:,0]))
    pcl_height = lidar_pcl_cpy[ind]
    _,ind = np.unique(pcl_height[:,0:2], axis=0 ,return_index=True)
    pcl_height = pcl_height[ind]
    height_map[np.int_(pcl_height[:,0]),np.int_(pcl_height[:,1])] = pcl_height[:,2]
    z_factor = 1/(configs.lim_z[1] - configs.lim_z[0])
    height_map = height_map * z_factor
    ## step 3 : temporarily visualize the height map using OpenCV to make sure that vehicles separate well from the background
    vis = False
    view_bev_img(height_map,vis=False,save_to_video=False,file_name='BEV_height_img_step2')
    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
    # lidar_pcl_cpy = []
    # lidar_pcl_top = []
    # height_map = []
    # intensity_map = []

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    lidar_pcl_top = pcl_height
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map
    
    test_map = np.zeros((configs.bev_height, configs.bev_width, 3))
    test_map[:, :, 0] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    test_map[:, :, 1] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    test_map[:, :, 2] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map
    # temporaly view stacked BEV map
    view_bev_img(test_map, vis=False,save_to_video=False,file_name='BEV_stacked_img_step2')

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map
    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    print(np.amin(bev_maps.data.tolist()),np.amax(bev_maps.data.tolist()))
    return input_bev_maps

def view_bev_img(bev_map,vis,save_to_video,file_name):
    if vis:
        img = bev_map * 256
        img = img.astype(np.uint8)
        while (1):
            cv2.imshow('bev_map', img)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        if save_to_video:
            print('test : ',read_global_var())
            save_image(img=img,file_name=file_name,dir='/home/jasoncat/Udacity/Final_Project/nd013-c2-fusion-starter/summary_related',index=read_global_var())
        cv2.destroyAllWindows()
