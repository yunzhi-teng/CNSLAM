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
struct DecomposedF{

};
struct DecomposedE {
    Eigen::Matrix<double,3,3> R;
    Eigen::Matrix<double,3,1> t;
};
DecomposedE decomposeE(const Eigen::Matrix<double,3,3> &E);
Eigen::Matrix<double,3,3> computeF(const std::vector<cv::DMatch>& matches);
DecomposedF decomposeF(const Eigen::Matrix<double,3,3> &F);
Eigen::Matrix<double,3,3> computeE(const std::vector<cv::Point2f> &points1,const std::vector<cv::Point2f> &points2, const Eigen::Matrix<double,3,3> K);