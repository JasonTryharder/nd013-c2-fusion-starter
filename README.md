# Lidar and camera fusion  
This project demonstrated a implementation of object detection using deep learning approach, and tracking using EKF to fuse lidar and camera detections  
[![label_vs_detected_object_Thumbnail.png](/summary_related/label_vs_detected_object_Thumbnail.png)](/summary_related/label_vs_detected_object.mp4)  
### The following diagram illustrates the API data flow, and steps makes up the detection and tracking functions  
![System_Detection_and_Tracking.png](/summary_related/System_Detection_and_Tracking.png)  
### For detection tasks, API will provide:  
1. data visulization  
Depend on the dataset/sensor set of choice, data format will vary, some exploratory analysis on the data is beneficial to understand the challenge of the project  
- note: Waymo utilizes multiple sensors including multiple types of Lidar   
	- x4 short range Perimeter Lidar, vertical -90 to +30 degree, 0-20M [link1](https://blog.waymo.com/2020/03/introducing-5th-generation-waymo-driver.html)  
	- x1 360 Lidar, vertical -17.6 to +2.4 degree, <70M [link2](https://waymo.com/intl/zh-cn/waymo-one/?loc=sf)  
![waymo_sensor_illustration.png](/summary_related/waymo_sensor_illustration.png)  
- view the range images  
	- This data structure holds 3d points as a 360 degree "photo" of the scanning environment with the row dimension denoting the elevation angle of the laser beam and the column dimension denoting the azimuth angle. With each incremental rotation around the z-axis, the lidar sensor returns a number of range and intensity measurements, which are then stored in the corresponding cells of the range image.  
	- In the figure below(credit: [udacity](https://classroom.udacity.com/nanodegrees/nd0013/parts/cd2690/modules/d3a07469-74b5-49c2-9c0e-3218c3ecd016/lessons/09368e69-a6e0-4109-b479-515cd7f5f518/concepts/0c8e77d9-163e-411d-a8fe-00cb3e40d7d0)), a point ***p*** in space is mapped into a range image cell, which is defined by azimuth angle ***alpha/yaw*** and inclination ***beta/pitch***, and inside each cell, it contains ***range,intensity,elongation and the vehicle pose***  
![range_img_udacity.png](/summary_related/range_img_udacity.png)  
video shows range and intensity channel vertically stacked  
[![range_img_step1_Thumbnail.png](/summary_related/range_img_step1_Thumbnail.png)](/summary_related/range_img_step1.mp4)

- view the pointcloud using open3d module  
With the help of spherical coordinates, also the extrinsic calibration of the top lidar, and transpose to vehicle coordinates, we can reconstruct x,y,z from range image
- By analyzing a few point cloud images, notice:   
	- Preceeding vehicles rear bumper receives most signals, features are most reliable in a frame by frame basis  
![show_pcl_20-54-51.png](/summary_related/show_pcl_20-54-51.png)   
	- Transparent objects like tail lights, windshields do not reflect well on lidar beams, features are not reliable in a frame by frame basis  
![show_pcl_20-51-34.png](/summary_related/show_pcl_20-51-34.png)  
	- Due to limited angular resolution, further away from the ego vehicle, the bind spots will increase, at a point, smaller objects like pedestrain, cyclist will be hidden in between  
	- Due to mounting position and viewing angle of the lidar, ego vehicle proximity presents a significant amount of blind spots, which will be addressed via perimeter lidar, this area's detection is important in making change lane maneuvers  
	- Lidar data also showed enough accuracy to differenciate lane sperations in the middle 
![show_pcl_20-52-50.png](/summary_related/show_pcl_20-52-50.png)  

2. data preprocessing  
Depend on the system architechture and the type of NN selected as detector, data collected from sensor(raw data) need to be processed to fit pipline, various operations including:  
	- crop the view to focus on predefined region   
	- map each individual channel of range image to 8bit data and threshold the object of interest to the middle part of dynamic range( by eliminating lidar registed data outliers)  
 	- convert range image(Waymo data format) to pcl( point cloud)  
Below shows intensity channel of ***cropped, 8bit*** image consist of the BEV image   
[![BEV_intensity_img_step2.mp4](/summary_related/BEV_intensity_img_step2_thumbnail.png)](/summary_related/BEV_intensity_img_step2.mp4)
Below shows height channel of ***cropped, 8bit*** image consist of the BEV image   
[![BEV_height_img_step2.mp4](/summary_related/BEV_height_img_step2_thumbnail.png)](/summary_related/BEV_height_img_step2.mp4)  
Notice height and intensity channel have different emphasis on the detected objects  
	- convert pcl to BEV(birds eye view) 
![BEV_stacked_img_step2.mp4](/summary_related/BEV_stacked_img_step2.mp4)
3. detector  
There is various processing pipeline for object detection and classification based on point-clouds. The pipeline structure summarized(credit: [Udacity](https://classroom.udacity.com/nanodegrees/nd0013/parts/cd2690/modules/d3a07469-74b5-49c2-9c0e-3218c3ecd016/lessons/cbe1917f-ffe4-4b8c-87b8-11edb85d79ff/concepts/39e780c5-e37e-43ba-9b48-dca8b4a67a7d)) consists of three major steps, which are   
	(1) data representation  
	(2) feature extraction   
	(3) model-based detection.  
Below [chart](https://classroom.udacity.com/nanodegrees/nd0013/parts/cd2690/modules/d3a07469-74b5-49c2-9c0e-3218c3ecd016/lessons/cbe1917f-ffe4-4b8c-87b8-11edb85d79ff/concepts/39e780c5-e37e-43ba-9b48-dca8b4a67a7d) is a data flow with pcl as input and classified and detected object as output  
![3d-objectDetection-pipeline.png](/summary_related/3d-objectDetection-pipeline.png)  
#### Step1 Data representation   
With the prevalence of convolutional neural networks (CNN) in object detection, point cloud representations are required to have a structure that suits the need of the CNN, so that convolution operations can be efficiently applied, the avaliable methods inlcude:  
- Point-based data representation : [PointNet](https://arxiv.org/abs/1612.00593), [PointNet++](https://arxiv.org/abs/1706.02413), [LaserNet](https://arxiv.org/abs/1903.08701)  
	- advantage: leave the structure of the point cloud intact so that no information is lost  
	- disadvantage: relatively high need for memory resources as a large number of points has to be transported through the processing pipeline  
- Voxel-based data representation : [VoxelNet](https://arxiv.org/abs/1711.06396)  
	- advantage: save memory resources as they reduce the number of elements that have to be held in memory simultaneously. Therefore, the feature extraction network will be computationally more efficient, because features are extracted for a group of voxels instead of extracting them for each point individually.   
- Pillar-based data representation : the point cloud is clustered not into cubic volume elements but instead into vertical columns rising up from the ground up. [PointPillars](https://arxiv.org/abs/1812.05784)  
	- advantage: Segmenting the point cloud into discrete volume elements saves memory resources  
- Frustum-based data representation : When combined with another sensor such as a camera, lidar point clouds can be clustered based on pre-detected 2d objects, such as vehicles or pedestrians. If the 2d region around the projection of an object on the image plane is known, a frustum can be projected into 3D space using both the internal and the external calibration of the camera. One method belonging to this class is e.g.[Frustum PointNets](https://arxiv.org/pdf/1711.08488v1.pdf). The following figure illustrates the principle.  
![Frustum_pointcloud_processing.png](/summary_related/Frustum_pointcloud_processing.png)
	- advantage: cluster lidar point using camera pre-detected region, noise reduction and increase accuracy, also save memory on saving point cloud   
	- disadvantage: requires a second sensor such as camera for pre-detection, however, camera been onboard selfdriving car is almost guaranteed   
- Projection-based data representation : While both voxel- and pillar-based algorithms cluster the point-cloud based on a spatial proximity measure, projection-based approaches reduce the dimensionality of the 3D point cloud along a specified dimension, there are three major approaches can be identified: front view (RV), range view (RV) and bird's eye view (BEV).  
BEV is the projection scheme most widely used. The reasons for this are three-fold: (1) The objects of interest are located on the same plane as the sensor-equipped vehicle with only little variance. Also, (2) the BEV projection preserves the physical size and the proximity relations between objects, separating them more clearly than with both the FV and the RV projection.   
#### Step 2 : Feature extraction
After the point cloud has been transformed into a suitable representation (such as a BEV projection), the next step is to identify suitable features. Currently, feature extraction is one of the most active research areas and significant progress has been made there in the last years, especially in improving the efficiency of the object detector models. The type of features that are most commonly used are (1) local, (2) global and (3) contextual features:  
	1. Local features, which are often referred to as low-level features are usually obtained in a very early processing stage and contain precise information e.g. about the localization of individual elements of the data representation structure.  
	2. Global features, which are also called high-level-features, often encode the geometric structure of an element within the data representation structure in relation to its neighbors.  
	3. Contextual features are extracted during the last stage of the processing pipeline. These features aim at being accurately located and having rich semantic information such as object class, bounding box shape and size and the orientation of the object.  
Some feature extractors that is found in the literatures:   
Point-wise feature extractors : PointNet uses the the entire point cloud as input. It extracts global structures from spatial features of each point within a subset of points in Euclidean space. But due to high memory requirements and computational complexity they are not yet suitable for use in autonomous driving  
Segment-wise feature extractors : The term "segment-wise" refers to the way how the point cloud is divided into spatial clusters (e.g. voxels, pillars or frustums). Once this has been done, a classification model is applied to each point of a segment to extract suitable volumetric features.One of the most-cited representatives of this class of feature extractors is VoxelNet. In a nutshell, the idea of [VoxelNet](https://arxiv.org/abs/1711.06396) is to encode each voxel via an architecture called "Voxel Feature Extractor (VFE)" and then combine local voxel features using 3D convolutional layers and then transform the point cloud into a high dimensional volumetric representation. Finally, a region proposal network processes the volumetric representation and outputs the actual detection results.[link](https://github.com/qianguih/voxelnet) to algorithm  
Convolutional Neural Networks (CNN): In recent years, many of the approaches for image-based object detection have been successfully transferred to point cloud processing. In most cases, the backbone networks used for image-based object detection can be directly used for point clouds as well. In order to balance between detection accuracy and efficiency, the type of backbones can be chosen between deeper and densely connected networks or lightweight variants with few connections.
#### Step 3 : Detection and Prediction Refinement
Once features have been extracted from the input data, a detection network is needed to generate contextual features (e.g. object class, bounding box) and finally output the model predictions. Depending on the architecture, the detection process can either perform a single-pass or a dual-pass. Based on the detector network architecture, the available types can be broadly organized into two classes, which are dual-stage encoders such as [R-CNN](https://arxiv.org/pdf/1311.2524.pdf), [Faster R-CNN](https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf) or [PointRCNN](https://arxiv.org/abs/1812.04244) or single-stage encoders such as [YOLO](https://arxiv.org/abs/1506.02640) or [SSD](https://arxiv.org/abs/1512.02325). In general, single-stage encoders are faster than dual-stage encoders, which makes them more suited for real-time applications such as autonomous driving.

This project uses [Complex Yolo](https://arxiv.org/abs/1803.06199) and [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D) for real time 3D object detection on point clouds,As can be seen from the following figure, the main pipeline of Complex YOLO consists of three steps:
	1. Transforming the point cloud into a bird's eye view (BEV)
	2. Complex YOLO on BEV map
	3. 3D bounding box re-conversion
One of the aspects that makes Complex YOLO special is the extension of the classical Grid RPN approach, which estimates only bounding box location and shape, by an orientation angle, which it encodes as a complex angle in Euler notation (hence the name "E-RPN") such that the orientation may be reconstructed as \mathrm{arctan2}(Im,Re)arctan2(Im,Re).
![complexYOLO_pipeline.png](/summary_related/complexYOLO_pipeline.png)
Below video demonstrates complex YOLO detection with labels(ground truth) drawn. Also note that in this project, we are only focussing on the detection of vehicles, even though the Waymo Open dataset contains labels for other road users as well.
[![label_vs_detected_object_Thumbnail.png](/summary_related/label_vs_detected_object_Thumbnail.png)](/summary_related/label_vs_detected_object.mp4)

4. Evaluating Object Detectors
Object detection algorithms need to perform two main tasks, which are
- to decide whether an object exists in the scene and
- to determine the position, the orientation and the shape of the object
Average Precision (AP) metric was proposed as a suitable metric for object detection. 
Several steps to implement the metrics :
	1. find pairings between ground-truth labels and detections by looking at label BB and detection BB IOUs, so that we can determine wether an object has been (a) missed (false negative), (b) successfully detected (true positive) or (c) has been falsely reported (false positive). 
	2. determine the number of false positives and false negatives for the current frame.And an overall performance after all frame have been processed 
	3. precision" and "recall" which are based on the accumulated number of positives and negatives from all frames will be calculated 
	4. The idea of the average precision (AP) metric is to compact the information within the precision-recall curve into a single number, which can be used to easily compare algorithms with each other. This goal is achieved by summing the precision values for different (=11) equally spaced recall values:(illustrated thru the following figure)
![AveragePrecision.png](/summary_related/AveragePrecision.png)
	5. Based on the observation that changing the IoU threshold affects both precision and recall, the idea of the mean average precision (mAP) measure is to compute the AP scores for various IoU thresholds and then computing the mean from these values. The following figure shows precision-recall curves for several settings of the IoU threshold
![mAP_illustration.png](/summary_related/mAP_illustration.png)
plot showing computed detection precision, recall, IOU, positional errors on 100 frames from training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord 
![computePR_stats.png](/summary_related/computePR_stats.png)
A test on ground truth is also performed to validate the evaluation pipeline:
Notice all metrics should be perfect since detector is by passed and using ground truth as detection result, but due to ***floating point precision***, some metrics shows some variation arround 1.  
![computePR_stats_use_label.png](/summary_related/computePR_stats_use_label.png)
### Future improvement 
This project implemented Darknet and Resnet-18 as backbone for feature extraction, and use BEV as data representation, future improvement can be using some other backbone while adding more detectable class. 
