# CNSLAM
CNSLAM stands for 3D vision Configurable framework Not only for SLAM.   
https://github.com/yunzhi-teng/CNSLAM/wiki
*\[WARNING\] There are lots of comments in the code without cleaning now, some for debugging, some for other function. Sorry for your reading. *
## dependency
*Each module differs*
- ceres for optimization
- pcl for point cloud visualization
- opencv
- eigen

## Module
For now, each module is an independent project.
https://github.com/yunzhi-teng/CNSLAM/wiki 
LVO, Tslam use opencv's interface. 
Other 3D vision algorithms are building from scratch (using linear algebra library eigen)
## to do
- Implement 3D vision related algorithms independently. Only use linear algebra library. 
- 3D vision System
## dataset used
https://github.com/yunzhi-teng/CNSLAM/wiki/Dataset-related
- kitti
- inria syntim
