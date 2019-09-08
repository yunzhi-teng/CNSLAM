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
#include <opencv2/core/eigen.hpp>
#include "../stereomatch/stereomatch.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "../triangulation/triangulation.hpp"
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


Eigen::MatrixXd pinv(Eigen::MatrixXd  A)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
    double  pinvtoler = 1.e-8; //tolerance
    int row = A.rows();
    int col = A.cols();
    int k = min(row,col);
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(col,row);
    Eigen::MatrixXd singularValues_inv = svd.singularValues();//singular value
    Eigen::MatrixXd singularValues_inv_mat = Eigen::MatrixXd::Zero(col, row);
    for (long i = 0; i<k; ++i) {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i) 
    {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }
    X=(svd.matrixV())*(singularValues_inv_mat)*(svd.matrixU().transpose());//X=VS+U*
 
    return X;
 
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
class PclWrapper {
public:
    PclWrapper(const int width_,const int height_,const string path_,const bool isdense) ;
    void savepcl();
    void addtopcl(double x, double y, double z);
    string path;
private:
   pcl::PointCloud<pcl::PointXYZ> cloud; 
   int pidx = 0;

};
PclWrapper::PclWrapper(const int width_,const int height_,const string path_,const bool isdense): path(path_)
{
    cloud.width = width_;
    cloud.height = height_;
    cloud.is_dense = isdense;
    cloud.points.resize(cloud.width *cloud.height);
}
void PclWrapper::savepcl()
{
    pcl::io::savePCDFileASCII(path,cloud);
}
void PclWrapper::addtopcl(double x, double y, double z)
{
    if(pidx < cloud.points.size())
    {
        cloud.points[pidx].x = x;
        cloud.points[pidx].y = y;
        cloud.points[pidx].z = z;
        // cout<<cloud.points[pidx]<<", ";
        pidx++;
    }
}
int main()
{



    cv::Mat img1 = cv::imread(path_1);
    cv::Mat img2 = cv::imread(path_2);
    // cv::imshow("img1",img1);
    // cv::waitKey(1);//l

    // cv::imshow("img2",img2);
    // cv::waitKey(-1);

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
    MatrixXd c1 = - (Po1.block(0,0,3,3)).inverse()*Po1.block<3,1>(0,3);//world coordinate
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
    cout<<"pn2: "<<Pn2<<endl;
    Matrix<double,3,3> T1,T2;
    T1 = Pn1.block<3,3>(0,0) * Po1.block<3,3>(0,0).inverse();
    T2 = Pn2.block<3,3>(0,0) * Po2.block<3,3>(0,0).inverse();
    cout<<"T1: "<<T1<<endl;
    cout<<"T2: "<<T2<<endl;
    cv::Mat cv_T1,cv_T2,dimg1,dimg2;
    cv::eigen2cv(T1,cv_T1);
    cv::eigen2cv(T2,cv_T2);
    cv::Size sz(768,576);
    cv::warpPerspective(img1,dimg1,cv_T1,sz);
    cv::warpPerspective(img2,dimg2,cv_T2,sz);
    // cv::perspectiveTransform(img1,dimg1,cv_T1);
    // cv::perspectiveTransform(img2,dimg2,cv_T2);

    //epipolar geometry verify
        Matrix<double,4,3> pn1inv;//right psuedo inverse
        pn1inv = pinv(Pn1);
    {
        int lineThickness = 2;
        cv::Point s,e,sp,s_manual,e_manual;
        Matrix<double,2,1> x;
        Matrix<double,3,1> x_h, xp1_h, epipole_h;
        Matrix<double ,4,1> c1_h;
        x_h << 200,200,1;
        sp.x = 200;//point in img1
        sp.y = 200;
        cout<< Pn2 <<pn1inv<<x_h<<endl;
        //find epipolar line 

        //1 project c1 and point to img2  -----> epipole and xp 
        Matrix<double,3,3> interm;
        interm = Pn2*pn1inv;
        cout<<"interm: "<<interm<<endl;
        xp1_h =  (interm * x_h);
        c1_h<< c1,1;
        epipole_h = Pn2 * c1_h;
        cout <<endl<<epipole_h<<";"<<xp1_h<<endl;
        //2 find a full line in image (not line segment) using P^2 geometry
        epipole_h(0,0) = epipole_h(0,0)/epipole_h(2,0);
        epipole_h(1,0) = epipole_h(1,0)/epipole_h(2,0);
        xp1_h(0,0) = xp1_h(0,0)/xp1_h(2,0);
        xp1_h(1,0) = xp1_h(1,0)/xp1_h(2,0);
        epipole_h(2,0) = 1;
        xp1_h(2,0) = 1;
        Matrix<double,3,1> line1, img_l,img_r,img_t,img_b, p1_h,p2_h;
        line1 = xp1_h.cross(epipole_h);
        img_l<<1, 0, 0;
        img_r<<1, 0, -dimg2.cols;
        p1_h =  line1.cross(img_l);
        p2_h =  line1.cross(img_r);

        s.x = p1_h(0,0)/p1_h(2,0);
        s.y = p1_h(1,0)/p1_h(2,0);
        e.x = p2_h(0,0)/p2_h(2,0);
        e.y = p2_h(1,0)/p2_h(2,0);
        cout <<endl<<s<<";"<<e<<endl;
        // cv::line(dimg2, s, e, cv::Scalar(100,255,100), lineThickness);
        // cv::circle(dimg1,sp,1,cv::Scalar(100,255,100),lineThickness);
    }

    cv::imshow("dimg1",dimg1);
    cv::waitKey(1);
    cv::imshow("dimg2",dimg2);
    cv::waitKey(1);

    // cv::imwrite("dimg1.png",dimg1);
    // cv::imwrite("img1.png",img1);
    // cv::imwrite("dimg2.png",dimg2);
    // cv::imwrite("img2.png",img2);

    // call stereo match
    {
        // cv::imshow("disp_l",ssd(dimg1,dimg2,"left"));
        // cv::imshow("disp_r",ssd(dimg1,dimg2,"right"));
        // cv::Mat dst = ssd(dimg1,dimg2,"left");

        // cv::Mat dstr = ssd(dimg1,dimg2,"right");
        // dstr = dstr*2;
        // dst = dst * 2;
        // cout<<"dst: "<<dst<<endl;
        // cv::imshow("disp_l",dst);
        // cv::imshow("dist_r", dstr);
        // cv::imwrite("disp_r.png",dstr);
        // cv::imwrite("disp_l.png",dst);

        // cv::imwrite("disp_r.png",ssd(dimg1,dimg2,"right"));

        // call triangulate
        {
            cv::Mat dst = cv::imread("disp_l.png",cv::IMREAD_GRAYSCALE);
            dst = dst /2;
            PclWrapper pclcloud(768,576,"sfm.pcd",true);
            vector<double> depths;
            for(int i = 0; i< dimg1.rows;i++)//left
            // for(int i = 0; i< 1;i++)//left
            {
                for(int j = 0; j< dimg1.cols;j++)
                // for(int j = 0; j< 1;j++)
                {
                    Matrix<double,3,1> x,x1;
                    Matrix<double,3,1> X;
                    // without kinv will give multiple plane projected
                    // 
                    Matrix<double,3,3> k1inv =  K1.inverse();
                    x(0,0) = j;
                    x(1,0) = i;
                    x(2,0) = 1;
                    // x = k1inv * x;
                    x1(0,0) = j - dst.at<uchar>(i,j);
                    x1(1,0) = i;
                    x1(2,0) = 1;
                    // x1 = k1inv * x1;

                    X = linear_triangulation(x,Pn1,x1,Pn2);
                    // cv::triangulatePoints(Pn1, Pn2, pts_1, pts_2, pts_4d);
                    if( abs(X(2,0)/1000) > 2.4 || abs(X(2,0)/1000)<2.15 && abs(X(2,0)/1000)>1.4)
                    {
                        pclcloud.addtopcl(-(X(0,0)/1000),-(X(1,0)/1000),-(X(2,0)/1000));
                    //     pclcloud.addtopcl(-X(0,0)/10e+5,-X(1,0)/10e+5,-X(2,0)/10e+5);
                        // depth.
                    }
                    

                }
            }
        //         {
        // double minVal,maxVal;
        // cv::minMaxLoc(depths, &minVal, &maxVal);
        // Mat tmp(240,320,CV_8UC3); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
        // for (unsigned int i=0; i   double _d = MAX(MIN((pointcloud[i].z-minVal)/(maxVal-minVal),1.0),0.0);
        //     circle(tmp, correspImg1Pt[i], 1, Scalar(255 * (1.0-(_d)),255,255), CV_FILLED);
        // }
        // cvtColor(tmp, tmp, CV_HSV2BGR);
        // imshow("out", tmp);
        // waitKey(0);
        // destroyWindow("out");
        //  }
            pclcloud.savepcl();
        }
    }
    cv::waitKey(-1);

}