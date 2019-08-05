#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/opencv.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
// #include <opencv2/sfm/numeric.hpp>
#include <fstream>
#include "projectionerr.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace cv;
using namespace std;
char *raw_path = "/data/SLAM/dataset/kitti/00/image_0/%06d.png";

#define Max 100
#define PI 3.14159265
#define Fi100 for (int i = 0 ;i <100; i++)
// Calculates rotation matrix given euler angles.
Mat eulerAnglesToRotationMatrix(Vec3f &theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
    // cout<<R_x<<endl;
    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
     
    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
     
     
    // Combined rotation matrix
    Mat R = R_z * R_y * R_x;
     
    return R;
 
}

Eigen::Matrix<double,3,1> inv_trans(Mat& R, Mat& t,Eigen::Matrix<double,3,1>& p)
{
    Eigen::Matrix<double,3,1> ps, t_e;
    Eigen::Matrix<double,3,3> Rinv_e;
    cv2eigen(R.inv(),Rinv_e);
    cv2eigen(t,t_e);
    ps = p - t_e;

    Eigen::Matrix<double,3,1> pr = Rinv_e*ps;
    return pr;

}

Eigen::Matrix<double,3,1> trans(Mat& R, Mat& t,Eigen::Matrix<double,3,1>& p)
{
    Eigen::Matrix<double,3,1> t_e;
    Eigen::Matrix<double,3,3> R_e;
    cv2eigen(R,R_e);
    cv2eigen(t,t_e);
    Eigen::Matrix<double,3,1> pr = R_e * p +t_e;
    return pr;
}


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

int main(){
    // Point3_<double> pw[10];//world coordinate
    // for (int i = 0 ; i< 10;i++)
    // {
    //     (pw[i]).x = 50 + i*3;
    //     pw[i].y = 60 + i*3;
    //     pw[i].z = 30 + i*3;
    // }
    Eigen::Matrix<double,3,1> pw[Max], p1[Max], p2[Max];//world coord ,c1 coord, c2 coord
    for (int i = 0 ;i <Max;i++)
    {
        pw[i](0,0) = 50 +i*3;
        pw[i](1,0) = 70 +i*3;
        pw[i](2,0) = 30 +i*3;

    }

    Vec3f theta;
    theta(0) = PI*60.0/180.0;
    theta(1) = PI*30.0/180.0;
    theta(2) = PI*60.0/180.0;
    Mat R =  eulerAnglesToRotationMatrix(theta);//apply to c1 coord------>world coord
    Mat t = (Mat_<double>(3,1) <<1,1,1) ;


    Vec3f theta1;
    theta1(0) = PI*45.0/180.0;
    theta1(1) = PI*45.0/180.0;
    theta1(2) = PI*30.0/180.0;
    Mat R1 =  eulerAnglesToRotationMatrix(theta1);//apply to c1 coord------>c2 coord
    Mat t1 = (Mat_<double>(3,1) <<5,2,1) ;

    for( int i = 0 ;i<Max ; i++)
    {
        p1[i] = inv_trans(R,t,pw[i]);
        p2[i] = trans(R1,t1,p1[i]);
    }

    cout<<"verify_3coordinate:"<< inv_trans(R1,t1,p2[0]) - p1[0]<<endl;
    cout<<"transform c1 to c2:"<<endl<<"R1:"<<R1<<endl<<"t1:"<<t1<<endl;
    //undistortion projection
    Mat K = (Mat_<double>(3, 3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);
    Eigen::Matrix<double,3,3> K_e;
    cv2eigen(K,K_e);

    Eigen::Matrix<double,3,1> p1_[Max], p2_[Max],p1uv[Max],p2uv[Max];//
    Fi100{
        p1_[i] = p1[i];
        p1_[i](0,0) = p1_[i](0,0)/p1_[i](2,0);
        p1_[i](1,0) = p1_[i](1,0)/p1_[i](2,0);
        p1_[i](2,0) = p1_[i](2,0)/p1_[i](2,0);
        p1uv[i] = K_e * p1_[i];

        p2_[i] = p2[i];
        p2_[i](0,0) = p2_[i](0,0)/p2_[i](2,0);
        p2_[i](1,0) = p2_[i](1,0)/p2_[i](2,0);
        p2_[i](2,0) = p2_[i](2,0)/p2_[i](2,0);
        
        p2uv[i] = K_e * p2_[i];
    }
    vector<Point2d> points1,points2;
    Fi100{
        Point2d point_;
        point_.x = p1uv[i](0,0);
        point_.y = p1uv[i](1,0);
        points1.push_back(point_);
    }
    Fi100{
        Point2d point_;
        point_.x = p2uv[i](0,0);
        point_.y = p2uv[i](1,0);
        points2.push_back(point_);
    }

    Mat E = findEssentialMat(points1,points2,K);
    cout<<"E_recover:"<<E<<endl;
    cout<<"E:"<< skew(t1)*R1 <<endl;
    Mat R_recover,t_recover,mask;
    recoverPose(E,points1,points2,K,R_recover,t_recover,mask);
    cout<<"recoveredtoE:"<<skew(t_recover)*R_recover<<endl;
    cout<<"R_recover:"<<R_recover<<endl;
    cout<<"t_recover:"<<t_recover<<endl;
    cout<<"p1:"<<p1[0]<<endl;
    cout<<"p2:"<<p2[0]<<endl;
    cout<<"p2-recovertrans:"<< trans(R_recover,t_recover,p1[0])<<endl;
    cout<<"mask:"<<mask<<endl;


    vector<Point3d> point_3D;
    // triangulation(keypoints1, keypoints2, matches, R, t, point_3D);


}