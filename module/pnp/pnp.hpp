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
struct Rt_pnp {
    Eigen::Matrix<double,3,3> R;
    Eigen::Matrix<double,3,1> t;
};
Rt_pnp solve_pnp(const Eigen::Matrix<double,3,3> &K, const std::vector<cv::Point2d> &pts_2d, 
                    const std::vector<cv::Point3d> &pts_3d);