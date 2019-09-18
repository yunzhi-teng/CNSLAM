# CNSLAM
CNSLAM stands for 3D vision Configurable framework Not only for SLAM.  
See **[Wiki](https://github.com/yunzhi-teng/CNSLAM/wiki)** for more information  
*\[WARNING\] There are lots of comments in the code without cleaning now, some for debugging, some for other function. Sorry for your reading.*
## Description
We separate CNSLAM's modules into three level.  
Level 3, **Pipeline** : LVO, Tslam  
Level 2, **Algorithm** : 3D vision algorithms from scratch (using linear algebra library eigen)  
Level 1, **Math** : numerical math methods(eg. matrix decompose) implemented by myself  

## Dependency
*Each module differs. For now,*  
*Level 3 : use opencv's interface.*  
*Level 2 : eigen and some with opencv's datastructure*  
*Level 1 : eigen's datastructure*  
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
5. Stereo match  
6. triangulate  
7. estimate essential matrix and recover pose
8. pnp  
benchmark  
See **[Wiki](https://github.com/yunzhi-teng/CNSLAM/wiki)** for more information  
## Usage
```py
git clone  
cd CNSLAM  
mkdir build && cd build && cmake ..  
make  
```
## Result 
See **[Wiki](https://github.com/yunzhi-teng/CNSLAM/wiki)** for each module's result  
## to do
- Implement 3D vision related algorithms independently.  
- 3D vision System
## Dataset used
See **[Dataset used](https://github.com/yunzhi-teng/CNSLAM/wiki/Dataset-related)**  for more information
- kitti
- inria syntim
