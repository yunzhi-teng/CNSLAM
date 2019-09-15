#include <iostream>
#include <string>
#include "fematrix.hpp"
using namespace cv;
using namespace std;
// char *path_1 = "/data/SLAM/dataset/syntim_stereo/Sport0_OG0.bmp";
// char *path_2 = "/data/SLAM/dataset/syntim_stereo/Sport1_OG0.bmp";
char *path_1 = "/data/SLAM/slambook/ch7/1.png";
char *path_2 = "/data/SLAM/slambook/ch7/2.png";
template<typename T>
Mat
skewMat( const Mat_<T> &x )
{
  Mat_<T> skew(3,3);
  skew <<   0 , -x(2),  x(1),
          x(2),    0 , -x(0),
         -x(1),  x(0),    0;

  return std::move(skew);
}

Mat
skew( InputArray _x )
{
  const Mat x = _x.getMat();
  const int depth = x.depth();
  CV_Assert( x.size() == Size(3,1) || x.size() == Size(1,3) );
  CV_Assert( depth == CV_32F || depth == CV_64F );

  Mat skewMatrix;
  if( depth == CV_32F )
  {
    skewMatrix = skewMat<float>(x);
  }
  else if( depth == CV_64F )
  {
    skewMatrix = skewMat<double>(x);
  }
  else
  {
    //CV_Error(CV_StsBadArg, "The DataType must be CV_32F or CV_64F");
  }

  return skewMatrix;
}
int main()
{
    cv::Mat img1 = cv::imread(path_1);
    cv::Mat img2 = cv::imread(path_2);
    Mat desc1, desc2;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    descriptor->compute(img1, keypoints1, desc1);
    descriptor->compute(img2, keypoints2, desc2);

    vector<DMatch> match;

    matcher->match(desc1, desc2, match);
    double min_dist = 8888, max_dist = 0;
    for (int i = 0; i < desc1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist < max_dist)
            max_dist = dist;
    }
    for (int i = 0; i < desc1.rows; i++)
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
    vector<Point2f> points1, points2;
    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
        Mat essential_mat;
    {//opencv findE and pose
        Mat R, t;
        // Mat K = (Mat_<double>(3, 3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);//00
        //seq 11
        // Mat K = (Mat_<double>(3, 3) <<7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02,0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02,0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);
        // Mat K = (Mat_<double>(3, 3) <<-933.506409, 0, 377.685547, 0, -907.118286, 287.696991, 0, 0, 1);
        Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
        Point2d principal_point( 325.1, 249.7);
        double focal_length = 521;
        Mat confi;
        essential_mat = findEssentialMat(points1, points2, focal_length, principal_point);
        cout << "E_cv:" << essential_mat << endl;
        recoverPose(essential_mat, points1, points2, R, t, focal_length, principal_point);
        cout << "t_cv:" << t.t() << endl;
        cout << "R_cv: " << R << endl;
        cout << "E_cv_t^R: "<<  skew(t)* R<<endl;
        Mat fundamentalmat;
        // fundamentalmat = findFundamentalMat()
    }

    {//findE
        Eigen::Matrix<double,3,3> K,E;
        // K << -933.506409, 0, 377.685547, 0, -907.118286, 287.696991, 0, 0, 1;
        K <<    520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;//tum
        E = computeE(points1,points2,K);
        DecomposedE Rt = decomposeE(E);

        cout<<endl<<endl<< "selffunc_cv_E : "<<endl;
        cv2eigen(essential_mat,E);
        DecomposedE Rt_cv = decomposeE(E);
    }


    return 0;
}