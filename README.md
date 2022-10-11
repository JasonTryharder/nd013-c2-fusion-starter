# Lidar and camera fusion  
This project demonstrated a implementation of object detection using a trained model using complex-yolo or  , and tracking using EKF to fuse lidar and camera detections  
[![label_vs_detected_object_Thumbnail.png](/summary_related/label_vs_detected_object_Thumbnail.png)](/summary_related/label_vs_detected_object.gif)  
### The following diagram illustrates the project data flow, and steps makes up the detection and tracking functions  
![System_Detection_and_Tracking.png](/summary_related/System_Detection_and_Tracking.png) 
![Tracking_flow_zoomin](/summary_related/EKF_flow.png) 
### Setup for local environment 
refer to instruction provided in this [link](https://github.com/JasonTryharder/nd013-c2-fusion-starter/blob/main/setup_README.md#installation-instructions-for-running-locally)
### Waymo AV sensor setup
note: Waymo utilizes multiple sensors including multiple types of Lidar   
	- x4 short range Perimeter Lidar, vertical -90 to +30 degree, 0-20M [link1](https://blog.waymo.com/2020/03/introducing-5th-generation-waymo-driver.html)  
	- x1 360 Lidar, vertical -17.6 to +2.4 degree, <70M [link2](https://waymo.com/intl/zh-cn/waymo-one/?loc=sf)  
![waymo_sensor_illustration.png](/summary_related/waymo_sensor_illustration.png)  
### Detection tasks:  
1.  Data visulization  
Depend on the dataset/sensor set of choice, data format will vary, some exploratory analysis on the data is beneficial to understand the challenge of the project  
  	1.1 View the range images  
    	- This data structure holds 3d points as a 360 degree "photo" of the scanning environment with the row dimension denoting the elevation angle of the laser beam and the column dimension denoting the azimuth angle. With each incremental rotation around the z-axis, the lidar sensor returns a number of range and intensity measurements, which are then stored in the corresponding cells of the range image.  
    	- In the figure below(credit: [udacity](https://classroom.udacity.com/nanodegrees/nd0013/parts/cd2690/modules/d3a07469-74b5-49c2-9c0e-3218c3ecd016/lessons/09368e69-a6e0-4109-b479-515cd7f5f518/concepts/0c8e77d9-163e-411d-a8fe-00cb3e40d7d0)), a point ***p*** in space is mapped into a range image cell, which is defined by azimuth angle ***alpha/yaw*** and inclination ***beta/pitch***, and inside each cell, it contains ***range,intensity,elongation and the vehicle pose***  
![range_img_udacity.png](/summary_related/range_img_udacity.png)  
img shows range and intensity channel vertically stacked  
[![range_img_step1_Thumbnail.png](/summary_related/range_img_step1_Thumbnail.png)](/summary_related/range_img_step1.gif)

	2.2 View the pointcloud using open3d module  
		With the help of spherical coordinates, also the extrinsic calibration of the top lidar, and transpose to vehicle coordinates, we can reconstruct x,y,z from range image
	2.3 By analyzing a few point cloud images, notice:   
    	- Preceeding vehicles rear bumper receives most signals, features are most reliable in a frame by frame basis  
![show_pcl_20-54-51.png](/summary_related/show_pcl_20-54-51.png)   
    	- Transparent objects like tail lights, windshields do not reflect well on lidar beams, features are not reliable in a frame by frame basis  
![show_pcl_20-51-34.png](/summary_related/show_pcl_20-51-34.png)  
    	- Due to limited angular resolution, further away from the ego vehicle, the bind spots will increase, at a point, smaller objects like pedestrain, cyclist will be hidden in between  
    	- Due to mounting position and viewing angle of the lidar, ego vehicle proximity presents a significant amount of blind spots, which will be addressed via perimeter lidar, this area's detection is important in making change lane maneuvers  
    	- Lidar data also showed enough accuracy to differenciate lane sperations in the middle 
![show_pcl_20-52-50.png](/summary_related/show_pcl_20-52-50.png)  

2. Data preprocessing  
Depend on the system architechture and the type of NN selected as detector, data collected from sensor(raw data) need to be processed to fit pipline, various operations including:  
	2.1 Crop the view to focus on predefined region   
	2.2 Map each individual channel of range image to 8bit data and threshold the object of interest to the middle part of dynamic range( by eliminating lidar registed data outliers)  
	2.3 Convert range image(Waymo data format) to pcl( point cloud)  
        - Below shows **intensity** channel of cropped, 8bit image consist of the BEV image   
[![BEV_intensity_img_step2](/summary_related/BEV_intensity_img_step2_thumbnail.png)](/summary_related/BEV_intensity_img_step2.gif)  
        - Below shows **height** channel of cropped, 8bit image consist of the BEV image   
[![BEV_height_img_step2](/summary_related/BEV_height_img_step2_thumbnail.png)](/summary_related/BEV_height_img_step2.gif)  
Notice height and intensity channel have different emphasis on the detected objects  
        - Convert pcl to BEV(birds eye view) 
[![BEV_stacked_img_step2](/summary_related/BEV_stacked_img_step2_thumbnail.png)](/summary_related/BEV_stacked_img_step2.gif)
1. Detector  
There is various processing pipeline for object detection and classification based on point-clouds. The pipeline structure summarized(credit: [Udacity](https://classroom.udacity.com/nanodegrees/nd0013/parts/cd2690/modules/d3a07469-74b5-49c2-9c0e-3218c3ecd016/lessons/cbe1917f-ffe4-4b8c-87b8-11edb85d79ff/concepts/39e780c5-e37e-43ba-9b48-dca8b4a67a7d)) consists of three major steps, which are   
	3.1 Data representation  
	3.2 Feature extraction   
	3.3 Model-based detection.  
Below [chart](https://classroom.udacity.com/nanodegrees/nd0013/parts/cd2690/modules/d3a07469-74b5-49c2-9c0e-3218c3ecd016/lessons/cbe1917f-ffe4-4b8c-87b8-11edb85d79ff/concepts/39e780c5-e37e-43ba-9b48-dca8b4a67a7d) is a data flow with pcl as input and classified and detected object as output  
![3d-objectDetection-pipeline.png](/summary_related/3d-objectDetection-pipeline.png)  
### Step 2 tracking tasks :
4. Single track tracking thru EKF 
Either Kalman filter or Extended-Kalman filter or Unsented Kalman filter all accomplish one goal: predict state estimate by a joint probability distribution for the states over each frame, at each frame it takes two step : prediction and measurement update.  
Specifically kalman filter will perform the following: 
	4.1 Calculate the time step **Delta t** and the new state transition matrix **F** and process noise covariance matrix **Q**.
![KF_equ](/summary_related/KF_prediction.png)
	4.2 Predict state and covariance to the **next timestamp**.
      	- Process noise is added based on driving situation, for example: high way driving and AEB(automatic emergency breaking) situations process noise will be different if we assume constant velocity, since the speed error will be much larger in AEB situation, updated prediction equ, factored in process noise nu
![KF_equ_more](/summary_related/KF_prediction_more.png) 
![KF_equ_more_1](/summary_related/KF_prediction_more_1.png) 
	4.3 Transform the state from **vehicle** to **sensor** coordinates.
   	    - Measurement update: 
![KF_equ](/summary_related/KF_measurement.png)
        - measurement equation for camera from a 6D(x,y,z,vx,vy,vz) vector to 2D(x,y) is non-linear 
![KF_measurement_camera_1](/summary_related/measurement_equ_camera_1.png)
	4.4 In case of a camera measurement, use the **nonlinear measurement model** and calculate the new Jacobian, otherwise use the **linear measurement model** to update state and covariance.
note: kalman filter assumes linear mapping matrix for the state transition matrix and measurement update matrix, EKF and UKF are ways to obtain a linear representation at non-linear situation, such as variant speed(with acceleration) or camera measurement model  
note: to get a linear representation of non-linear equation, we used multivariant taylor expansion(first order), so there is a Jacobian matrix is needed for first exapnsion  
![KF_measurement_Taylo](/summary_related/measurement_equ_Taylor_0.png)
        - Below is a setup for 2x6 Jacobian
![KF_measurement_Taylo](/summary_related/measurement_equ_Taylor_0.png)
![KF_measurement_Taylo](/summary_related/measurement_equ_Taylor_0.png)
        - KF parameter definition: 
![KF_equ](/summary_related/KF_Definition.png)


5. Track Management:
A multi-target tracking system has to fulfill the following tasks in addition to a single-target tracking:
	5.1 Initialize new tracks
      	- Before measurement can be tracked against tracks, track has to be initialized first, depending on first received measurement been camera or lidar, choice is at engineer's hand how to use populate the tracks, eg wait several measurement to initialize nad can get velocity as well or, disgard camera and wait for lidar so state has distance
![track_mgr_init_0](/summary_related/track_mgr_init_0.png)
    	- Covariance matrix also need to be initialized, based on sensor to vehicle transformation, also error term should also reflect the initial state, such as velocity estimation error can be larger to reflect the lack of velocity measurement update 
![track_mgr_init_1](/summary_related/track_mgr_init_1.png)
	5.2 Delete old tracks
		- Tack management should also be able to delet old tracks that their score is below certain threshold to stop tracking
	5.3 Assign some confidence value to a track
     	- A track scoring system can help to keep track of tracks and provide metrics(confidence) for track deletion, many heuristic method is implemented here. approach implemented in this project is detection in the last 6 frames over number of frames(6), and a state name("initialized", "tentative", "confirmed") is assigned based on score and wether the track is new or not, and when to delete a track, this extra information is not mandantory but it can help the track management keep better track of things 
![track_mgr_init_1](/summary_related/track_mgr_score.png)
6. Track/measurement Association:
	6.1 Associate measurements to tracks
      	- Association handels which measurement to update with which track, it assumes each track originates from at most one track, and each track generate at most one measurement  
![track_mgr_init_1](/summary_related/association_MHD_0.png)
        - This project uses Mahalanobis distance(MHD) to measure association, it incorprates the residual gamma(prediction state - measurement) and inverse of residual covariance S which is process covariance P transformed to maeasurement space plus measurement noise. so the smaller uncertainty result in less distance, more discussion about advanced association method is discussed in later section
![track_mgr_init_1](/summary_related/association_MHD_0.png) 
	6.2 visibility checks per each sensor's FOV 
      	- different sensor has differnet FOVs and to prevent score occilating due to situations where track is lack of measurement update due to out of FOV, the track management makes visibilty reasoning each track get updated, so for example when track is outside of camera FOV and track is not updated with maeasurement, the score is not going to be reduced.(probabilty reasoning to decide visibilty and also uses dynamic occolusion reasoning to handle dynamic situation)   
![track_mgr_init_1](/summary_related/association_FOV.png)
	6.3 Probability based gating
      	- To further reduce the complexity to association, this project implemented **gating**, a measurement lies inside a track's gate if the Mahalanobis distance is smaller than the threshold calculated from the inverse cumulative **X^2(Chi-squared)** distribution. 
![track_mgr_init_1](/summary_related/association_gating.png)

## Evaluation
7. Object detection algorithms need to perform two main tasks, which are
- to decide whether an object exists in the scene and
- to determine the position, the orientation and the shape of the object
Average Precision (AP) metric was proposed as a suitable metric for object detection. 
Several steps to implement the metrics :
	7.1 Find pairings between ground-truth labels and detections by looking at label BB and detection BB IOUs, so that we can determine wether an object has been (a) missed (false negative), (b) successfully detected (true positive) or (c) has been falsely reported (false positive). 
	7.2 Determine the number of false positives and false negatives for the current frame.And an overall performance after all frame have been processed 
	7.3 Precision" and "recall" which are based on the accumulated number of positives and negatives from all frames will be calculated 
	7.4 The idea of the average precision (AP) metric is to compact the information within the precision-recall curve into a single number, which can be used to easily compare algorithms with each other. This goal is achieved by summing the precision values for different (=11) equally spaced recall values:(illustrated thru the following figure)
![AveragePrecision.png](/summary_related/AveragePrecision.png)
	7.5 Based on the observation that changing the IoU threshold affects both precision and recall, the idea of the mean average precision (mAP) measure is to compute the AP scores for various IoU thresholds and then computing the mean from these values. The following figure shows precision-recall curves for several settings of the IoU threshold
![mAP_illustration.png](/summary_related/mAP_illustration.png)
	7.6 Plot showing computed detection precision, recall, IOU, positional errors on 100 frames from training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord 
![computePR_stats.png](/summary_related/computePR_stats.png)
	7.7 A test on ground truth is also performed to validate the evaluation pipeline:
Notice all metrics should be perfect since detector is by passed and using ground truth as detection result, but due to ***floating point precision***, some metrics shows some variation arround 1.  
![computePR_stats_use_label.png](/summary_related/computePR_stats_use_label.png)
	7.8 Below video demonstrates complex YOLO detection with labels(ground truth) drawn. Also note that in this project, we are only focussing on the detection of vehicles, even though the Waymo Open dataset contains labels for other road users as well.  
![label_vs_detected_object_Thumbnail](/summary_related/label_vs_detected_object.gif)

1. Tracking algorithm is tested using pre-recorded frames using **fpn_resnet only**, due to darknet model does not provide hight label, two test result is plotted: 
	8.1 Use camera and lidar detection together, one can observe there are three tracks get consistent detection and tracking, the detection accuracy measured using RMSE can stay below 0.4 all the time, this definetly will help increase tracking robustness
![RMSE_camera_lidar](/summary_related/RMSE_with_camera.png)
	8.2 Use lidar alone the tracking algorithm still showed strong robustness, it tracks the same 3 tracks only with slight higher RMSE error, however, using extra sensor(lidar) is strongly preferred, since with more sensor can increase tracking capability for other object such as human, cyclist, animals. also, camera can provide more contextual information such as sementic infomation
![RMSE_camera_lidar](/summary_related/RMSE_without_camera.png)
	8.3 Below video shows tracking algorithms, with left showing detection ground thruth, in the testing scenario, the vehicle is driving under realtive constant speed, which is similar to the motion model estimation, but it will face challenge in an AEB situation.
![Tracking_video](/summary_related/Final-my-tracking-results.gif)
	8.4 The motion model assumes no limitation in terms of direction a car can drive, which in real scenario, the car can only drive in some limitted directions, by update with bicycle model, and constant acceleration motion model, the tracking performance can be improved further

## Additional information  
9. Data representation   
With the prevalence of convolutional neural networks (CNN) in object detection, point cloud representations are required to have a structure that suits the need of the CNN, so that convolution operations can be efficiently applied, the avaliable methods inlcude:  
	9.1 Point-based data representation : [PointNet](https://arxiv.org/abs/1612.00593), [PointNet++](https://arxiv.org/abs/1706.02413), [LaserNet](https://arxiv.org/abs/1903.08701)  
      	- advantage: leave the structure of the point cloud intact so that no information is lost  
      	- disadvantage: relatively high need for memory resources as a large number of points has to be transported through the processing pipeline  
	9.2 Voxel-based data representation : [VoxelNet](https://arxiv.org/abs/1711.06396)  
      	- advantage: save memory resources as they reduce the number of elements that have to be held in memory simultaneously. Therefore, the feature extraction network will be computationally more efficient, because features are extracted for a group of voxels instead of extracting them for each point individually.   
	9.3 Pillar-based data representation : the point cloud is clustered not into cubic volume elements but instead into vertical columns rising up from the ground up. [PointPillars](https://arxiv.org/abs/1812.05784)  
      	- advantage: Segmenting the point cloud into discrete volume elements saves memory resources  
	9.4 Frustum-based data representation : When combined with another sensor such as a camera, lidar point clouds can be clustered based on pre-detected 2d objects, such as vehicles or pedestrians. If the 2d region around the projection of an object on the image plane is known, a frustum can be projected into 3D space using both the internal and the external calibration of the camera. One method belonging to this class is e.g.[Frustum PointNets](https://arxiv.org/pdf/1711.08488v1.pdf). The following figure illustrates the principle.  
![Frustum_pointcloud_processing.png](/summary_related/Frustum_pointcloud_processing.png)  
      	- advantage: cluster lidar point using camera pre-detected region, noise reduction and increase accuracy, also save memory on saving point cloud   
      	- disadvantage: requires a second sensor such as camera for pre-detection, however, camera been onboard selfdriving car is almost guaranteed   
	9.5 Projection-based data representation : While both voxel- and pillar-based algorithms cluster the point-cloud based on a spatial proximity measure, projection-based approaches reduce the dimensionality of the 3D point cloud along a specified dimension, there are three major approaches can be identified: front view (RV), range view (RV) and bird's eye view (BEV).  
        - BEV is the projection scheme most widely used. The reasons for this are three-fold: (1) The objects of interest are located on the same plane as the sensor-equipped vehicle with only little variance. Also, (2) the BEV projection preserves the physical size and the proximity relations between objects, separating them more clearly than with both the FV and the RV projection.   
10. Feature extraction
After the point cloud has been transformed into a suitable representation (such as a BEV projection), the next step is to identify suitable features. Currently, feature extraction is one of the most active research areas and significant progress has been made there in the last years, especially in improving the efficiency of the object detector models. The type of features that are most commonly used are (1) local, (2) global and (3) contextual features:  
	10.1 Local features, which are often referred to as low-level features are usually obtained in a very early processing stage and contain precise information e.g. about the localization of individual elements of the data representation structure.  
	10.2 Global features, which are also called high-level-features, often encode the geometric structure of an element within the data representation structure in relation to its neighbors.  
	10.3 Contextual features are extracted during the last stage of the processing pipeline. These features aim at being accurately located and having rich semantic information such as object class, bounding box shape and size and the orientation of the object.  
11. Feature extractors that are found in the literatures:   
	11.1 Point-wise feature extractors : PointNet uses the the entire point cloud as input. It extracts global structures from spatial features of each point within a subset of points in Euclidean space. But due to high memory requirements and computational complexity they are not yet suitable for use in autonomous driving  
	11.2 Segment-wise feature extractors : The term "segment-wise" refers to the way how the point cloud is divided into spatial clusters (e.g. voxels, pillars or frustums). Once this has been done, a classification model is applied to each point of a segment to extract suitable volumetric features.One of the most-cited representatives of this class of feature extractors is VoxelNet. In a nutshell, the idea of [VoxelNet](https://arxiv.org/abs/1711.06396) is to encode each voxel via an architecture called "Voxel Feature Extractor (VFE)" and then combine local voxel features using 3D convolutional layers and then transform the point cloud into a high dimensional volumetric representation. Finally, a region proposal network processes the volumetric representation and outputs the actual detection results.[link](https://github.com/qianguih/voxelnet) to algorithm  
	11.3 Convolutional Neural Networks (CNN): In recent years, many of the approaches for image-based object detection have been successfully transferred to point cloud processing. In most cases, the backbone networks used for image-based object detection can be directly used for point clouds as well. In order to balance between detection accuracy and efficiency, the type of backbones can be chosen between deeper and densely connected networks or lightweight variants with few connections.
12. Detection and Prediction Refinement
Once features have been extracted from the input data, a detection network is needed to generate contextual features (e.g. object class, bounding box) and finally output the model predictions. Depending on the architecture, the detection process can either perform a single-pass or a dual-pass. Based on the detector network architecture, the available types can be broadly organized into two classes, which are dual-stage encoders such as [R-CNN](https://arxiv.org/pdf/1311.2524.pdf), [Faster R-CNN](https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf) or [PointRCNN](https://arxiv.org/abs/1812.04244) or single-stage encoders such as [YOLO](https://arxiv.org/abs/1506.02640) or [SSD](https://arxiv.org/abs/1512.02325). In general, single-stage encoders are faster than dual-stage encoders, which makes them more suited for real-time applications such as autonomous driving.  
This project uses [Complex Yolo](https://arxiv.org/abs/1803.06199) and [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D) for real time 3D object detection on point clouds,As can be seen from the following figure, the main pipeline of Complex YOLO consists of three steps:  
	12.1 Transforming the point cloud into a bird's eye view (BEV)  
	12.2 Complex YOLO on BEV map  
	12.3 3D bounding box re-conversion  
One of the aspects that makes Complex YOLO special is the extension of the classical Grid RPN approach, which estimates only bounding box location and shape, by an orientation angle, which it encodes as a complex angle in Euler notation (hence the name "E-RPN") such that the orientation may be reconstructed as \mathrm{arctan2}(Im,Re)arctan2(Im,Re).  
![complexYOLO_pipeline.png](/summary_related/complexYOLO_pipeline.png)  
  
### Future improvement 
This project implemented Darknet and Resnet-18 as backbone for feature extraction, and use BEV as data representation, future improvement can be using some other backbone while adding more detectable class. 
