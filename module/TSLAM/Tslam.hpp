#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/opencv.hpp"
#include <fstream>
#include "projectionerr.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace cv;
using namespace std;
#define INTERVAL 2
#define INITIAL_INTER 3

class Keyframe
{
public:
    Mat R,t;
    vector<KeyPoint> keypoint;
    Mat desc;
    int id;
};
class Frameproccessing
{
public:
    Keyframe keyframe_;
    vector<int> matchkp_mp;
};
// class Mappoint_info
// {
// public:
//     vector<Keyframe> visible_keyframe;
//     vector<KeyPoint> keyframe_assoc_point;
// };
class Mappoint
{
public:
    // Mappoint_info info;
    vector<int> visible_keyframe_idx;//each element is a idx in allkeyframe
    vector<int> keyframe_indexed_pt2d;//in keypoint,,,same size as visible_kf_idx,,,same index [j] as visible_frame_idx
    Point3d pt3d;
    int id;
};


class Tslam
{
public:
    void initialize();
    void addframe();
    void ba();
    vector<Keyframe> all_keyframe;
    vector<Mappoint> all_mappoint;
    Frameproccessing last_frame;
    int image_index= 0;
private:
//00
    Mat K = (Mat_<double>(3, 3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);
//seq 11
    // Mat K = (Mat_<double>(3, 3) <<7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02,0.000000000000e+00 ,7.070912000000e+02, 1.831104000000e+02,0.000000000000e+00 ,0.000000000000e+00, 1.000000000000e+00);

};
