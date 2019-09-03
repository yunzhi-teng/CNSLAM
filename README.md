# CNSLAM
CNSLAM stands for 3D vision Configurable framework Not only for SLAM.  
See **[Wiki](https://github.com/yunzhi-teng/CNSLAM/wiki)** for more information  
*\[WARNING\] There are lots of comments in the code without cleaning now, some for debugging, some for other function. Sorry for your reading.*
## Dependency
*Each module differs*  
*LVO, Tslam use opencv's interface.*  
*Other 3D vision algorithms are building from scratch (using linear algebra library eigen)*  
- ceres for optimization
- pcl for point cloud visualization
- opencv
- eigen

## Module
*For now, each module is an independent project.*  
1. LVO---lightweight visual odometry  
2. Tslam---Toy Slam  
3. image_rectification and epipolar geometry
4. QR_decompose  
benchmark  
See **[Wiki](https://github.com/yunzhi-teng/CNSLAM/wiki)** for more information  
## Result 
See **[Wiki](https://github.com/yunzhi-teng/CNSLAM/wiki)** for each module's result  
## to do
- Implement 3D vision related algorithms independently. Only use linear algebra library. 
- 3D vision System
## Dataset used
See **[Dataset used](https://github.com/yunzhi-teng/CNSLAM/wiki/Dataset-related)**  for more information
- kitti
- inria syntim
