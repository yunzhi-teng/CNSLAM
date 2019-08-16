# CNSLAM
CNSLAM stands for Configurable framework Not only for SLAM,  
But now it is just a kid, maybe called "Characteristically Naive Slam"  
\[WARNING\] There are lots of comments in the code without cleaning now, some for debugging, some for other function. Good for learning, Sorry for your reading. 
## dependency
- ceres for optimization
- pcl for point cloud visualization
- opencv
- eigen

## Module
For now, each module is an independent project. 
### 1 LVO
Ligntweight Visual Odometry (monocular)
#### Feature
- estimate pose only according to two adjacent frame(2D--2D)
- triangulate
- per frame BA
#### pipeline
- get frame
- extract feature
- feature match with last frame
- find essential matrix based on epipolar geometry using the feature match
- estimate R and t by decomposing essential matrix
- compute camera's world coordinate : Now the R and t are relative transformation between the two frames. We need to derive where the camera is in world coordinate system.
- \[optional\] triangulate and per frame bundle adjustment
### 2 Tslam
It is a naive slam project (monocular)
#### Feature
- estimate pose using pnp(3D--2D)
- triangulate new 3D map point
- BA(for each frame or all frames or choosed frames)
#### Data structure
- Tslam: 
- Mappoint:
- Keyframe:
- Frameprocessing: including a Keyframe and a map from 2d keypoint in this keyframe to mappoint 
#### pipeline
- initialize
- addframe
### 3 benchmark
It is an environment for verifying 3D vision algorithm. 
#### Feature:
- manually created 3D points.
- manually rotate and translate
- manually projected to 2D image
- using the data above, verify PCP, epipolar geometry, BA, etc. 

## to do
- Now Opencv functions are intensively used. I will reimplement some of these for better analysis and improvement for the 3DV algorithms and systems. 
## dataset used
- kitti
