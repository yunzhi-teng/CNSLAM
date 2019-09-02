#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <opencv2/core/eigen.hpp>
using namespace Eigen;
using namespace std;

Matrix<double,3,1> t1;
Matrix<double,3,1> t2;
Matrix<double,3,3> R1;
Matrix<double,3,3> R2;
Matrix<double,3,3> K1;
Matrix<double,3,3> K2;
Matrix<double,3,4> Po1,Pn1;
Matrix<double,3,4> Po2,Pn2;
char *path_1 = "/data/SLAM/dataset/syntim_stereo/Sport0_OG0.bmp";
char *path_2 = "/data/SLAM/dataset/syntim_stereo/Sport1_OG0.bmp";

void Trectify(const Matrix<double,3,4>& po1,const Matrix<double,3,4>& po2,
             Matrix<double,3,4>& pn1, Matrix<double,3,4>& pn2, Matrix<double,3,3>& T1,Matrix<double,3,3>& T2)
{

}
void factorizeKRt(const Matrix<double,3,4>& p, Matrix<double,3,3>& R, Matrix<double,3,1>& t, Matrix<double,3,3>& K)
{
    Matrix<double,3,3> Q_ = p.block(0,0,3,3).inverse();
    // ColPivHouseholderQR<MatrixXd> qr(Q_);
    HouseholderQR<MatrixXd> qr(Q_);
    cout<<"qr_orig: "<<Q_<<endl;
    qr.compute(Q_);
    Matrix<double,3,3> Q = qr.householderQ();
    cout << "qr_Q: "<<Q<<endl;
    Matrix<double,3,3> R_ =  qr.matrixQR().template triangularView<Upper>();
    cout << "qr_R: "<<R_<<endl;
    cout<<"veri_qr: "<<Q*R_<<endl;

    

    // cout << "R_: "<<R_<<endl;
    R = Q.inverse();
    cout<<"fact_R: "<<R<<endl;
    t = R_*p.block<3,1>(0,3);
    cout<<"face_t: "<<t<<endl;
    K = R_.inverse();
    K = K /K(2,2);
    cout<<"fact_K: "<<K<<endl;

}

int main()
{



    cv::Mat img1 = cv::imread(path_1);
    cv::Mat img2 = cv::imread(path_2);
    cv::imshow("img1",img1);
    cv::waitKey(1);//l

    cv::imshow("img2",img2);
    cv::waitKey(-1);

    Po1<< 976.5, 53.82, -239.8, 3.875e+05, 98.49, 933.3, 157.4, 2.428e+05, 0.5790, 0.1108, 0.8077, 1118;
    cout<<"po1:" << Po1<<endl;
    Po2<< 976.7, 53.76, -240.0, 4.003e+04, 98.68, 931.0, 156.7, 2.517e+05, 0.5766, 0.1141, 0.8089, 1174;
    cout<<"po2:" << Po2<<endl;
    Matrix<double,3,3> R1,K1,R2,K2;
    Matrix<double,3,1> t1,t2;
    factorizeKRt(Po1,R1,t1,K1);
    factorizeKRt(Po2,R2,t2,K2);
    {//sport 1 l, 2 r

        Matrix<double,3,1> tc1,tt1;
        tc1 << -623.831787, -37.058510, -932.469971;//<< row by row

        Matrix<double,3,3> tR1_,tR1,tK1;
        Matrix<double,3,4> tPo1;
        tR1_ << 0.811845, 0.012814, -0.583732, -0.075058, 0.993754, -0.082574, 0.579028, 0.110851, 0.807736;
        tR1 = tR1_.transpose();

        //
        tR1 = R1;
        cout<< "R1i: "<<tR1<<endl;

        cout<< "c1i: "<<tc1<<endl;
        
        tK1 << -933.506409, 0, 377.685547, 0, -907.118286, 287.696991, 0, 0, 1;
        // K1(0,2) = K1(0,2)+160;
        cout<< "k1i: "<<tK1<<endl;

        tPo1.col(0) = tR1.col(0);
        tPo1.col(1) = tR1.col(1);
        tPo1.col(2) = tR1.col(2);
        tPo1.col(3) = -tR1*tc1;
        cout<<"po1i:" << tPo1<<endl;
        tPo1 = tK1*tPo1;
        cout<<"po1i:" << tPo1<<endl;
    }
    K1(0,2) = K1(0,2)+160;
    MatrixXd c1 = - (Po1.block(0,0,3,3)).inverse()*Po1.block<3,1>(0,3);
    MatrixXd c2 = - (Po2.block(0,0,3,3)).inverse()*Po2.block<3,1>(0,3);
    cout << "c1； "<<c1<<endl;
    Matrix<double,3,1> v1,v2,v3;
    v1 = c1 - c2;
    // cout<<"v1: "<<v1<<endl;
    // cout<< "deb_1: "<<R1.block<1,3>(2,0).transpose()<<endl;
    // cout<< "deb_2: "<<R1.block<1,3>(2,0).transpose()<<endl;
    v2 = R1.block<1,3>(2,0).transpose().cross(v1);

    // cout<<"v2: "<<v2<<endl;
    v3 = v1.cross(v2);
    // cout<<"v3: "<<v3<<endl;
    Matrix<double,3,3> R_n;
    R_n << v1.transpose()/v1.norm(), v2.transpose()/v2.norm(), v3.transpose()/v3.norm();
    // cout<<"R_n: "<<R_n<<endl;
    K1(0,1) = 0;
    Matrix<double,3,4> interm_1, interm_2;
    interm_1.col(0) = R_n.col(0);
    interm_1.col(1) = R_n.col(1);
    interm_1.col(2) = R_n.col(2);
    interm_1.col(3) = -R_n*c1;
    interm_2.col(0) = R_n.col(0);
    interm_2.col(1) = R_n.col(1);
    interm_2.col(2) = R_n.col(2);
    interm_2.col(3) = -R_n*c2;
    // interm_2 << R_n, -R_n*c2;

    Pn1 = K1 * interm_1;
    Pn2 = K1 * interm_2;
    cout<<"pn1: "<<Pn1<<endl;
    Matrix<double,3,3> T1,T2;
    T1 = Pn1.block<3,3>(0,0) * Po1.block<3,3>(0,0).inverse();
    T2 = Pn1.block<3,3>(0,0) * Po1.block<3,3>(0,0).inverse();
    cout<<"T1: "<<T1<<endl;
    cv::Mat cv_T1,cv_T2,dimg1,dimg2;
    cv::eigen2cv(T1,cv_T1);
    cv::eigen2cv(T2,cv_T2);
    cv::Size sz(768,576);
    cv::warpPerspective(img1,dimg1,cv_T1,sz);
    cv::warpPerspective(img2,dimg2,cv_T2,sz);
    // cv::perspectiveTransform(img1,dimg1,cv_T1);
    // cv::perspectiveTransform(img2,dimg2,cv_T2);
    cv::imshow("dimg1",dimg1);
    cv::waitKey(1);
    cv::imshow("dimg2",dimg2);
    cv::waitKey(-1);

}