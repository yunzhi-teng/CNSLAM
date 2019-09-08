#include <iostream>
// #include <ceres/ceres.h>
// #include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <opencv2/core/eigen.hpp>
Eigen::Matrix<double, 3,1> linear_triangulation(const Eigen::Matrix<double,3,1> &x,const Eigen::Matrix<double, 3,4> &P,const Eigen::Matrix<double,3,1> &x1,const Eigen::Matrix<double,3,4> &P1);