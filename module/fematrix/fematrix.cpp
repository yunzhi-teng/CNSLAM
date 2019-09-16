#include "fematrix.hpp"
#include <math.h>
inline double getx(const cv::Point2f &p)
{
    return (double)(p.x);
}
inline double gety(const cv::Point2f &p)
{
    return (double)(p.y);
}
static Eigen::Matrix<double,3,3> skews(const Eigen::Matrix<double,3,1>& in)
{
    Eigen::Matrix<double,3,3> s ;
    s<< 0,-in(2,0),in(1,0),in(2,0),0,-in(0,0),-in(1,0),in(0,0),0;
    return s;

}
Eigen::Matrix<double,3,3> computeF(const std::vector<cv::DMatch>& matches);
DecomposedF decomposeF(const Eigen::Matrix<double,3,3> &F);
DecomposedE decomposeE(const Eigen::Matrix<double,3,3> &E)
{
    DecomposedE res;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double,3,3> W;
    W << 0,-1,0,   1,0,0,   0,0,1;
    res.R = svd.matrixU() * W * svd.matrixV().transpose();
    res.t = svd.matrixU().block<3,1>(0,2);
    std::cout << "R_self: "<< res.R<<std::endl << svd.matrixU() * W.transpose() * svd.matrixV().transpose()<<std::endl;
    std::cout << "t_self: "<< (svd.matrixU() * W.transpose() * svd.matrixV().transpose()).inverse() * res.t<<std::endl;
    std::cout << "t^R_self: "<< skews(res.t) * res.R<<std::endl;


    return res;
    
}
Eigen::Matrix<double,3,3> computeE(const std::vector<cv::Point2f> &points1,
                                    const std::vector<cv::Point2f> &points2, const Eigen::Matrix<double,3,3> K)
{
    std::vector<Eigen::Matrix<double,3,1>> pts1_hat,pts2_hat;
    for(int i = 0; i< points1.size();i++)
    {
        Eigen::Matrix<double,3,1> pt1,pt2;
        pt1 << getx(points1[i]), gety(points1[i]), 1;
        pt2 << getx(points2[i]), gety(points2[i]), 1;
        pt1 = K.inverse() * pt1;
        // pt1 = pt1 / pt1(2);
        pt2 = K.inverse() * pt2;
        // pt2 = pt2 / pt2(2);
        pts1_hat.push_back(pt1);
        pts2_hat.push_back(pt2);
    }
    // x1~T E x2~ = 0
    Eigen::Matrix<double,9,1> e;
    Eigen::Matrix<double,8,9> A;
    int mapi2ptidx[8];
    for(int i = 0; i < 8 ; i++)
    {
        mapi2ptidx[i] = (1 + i*(points1.size()/8)) % points1.size();
    }
    for(int i = 0; i < 8 ; i++)
    {
        Eigen::Matrix<double,1,9> arow;
        std::cout << "pt1_"<<i<<": "<< pts1_hat[mapi2ptidx[i]]<<std::endl;
        std::cout << "pt2_"<<i<<": "<< pts2_hat[mapi2ptidx[i]]<<std::endl;
        arow<< pts2_hat[mapi2ptidx[i]](0) * pts1_hat[mapi2ptidx[i]](0), pts2_hat[mapi2ptidx[i]](0) * pts1_hat[mapi2ptidx[i]](1),
                pts2_hat[mapi2ptidx[i]](0) *pts1_hat[mapi2ptidx[i]](2), pts2_hat[mapi2ptidx[i]](1) * pts1_hat[mapi2ptidx[i]](0),
                pts2_hat[mapi2ptidx[i]](1) * pts1_hat[mapi2ptidx[i]](1), pts2_hat[mapi2ptidx[i]](1) * pts1_hat[mapi2ptidx[i]](2),
                pts2_hat[mapi2ptidx[i]](2) * pts1_hat[mapi2ptidx[i]](0), pts2_hat[mapi2ptidx[i]](2) * pts1_hat[mapi2ptidx[i]](1),
                pts2_hat[mapi2ptidx[i]](2) * pts1_hat[mapi2ptidx[i]](2);
        A.block<1,9>(i,0) = arow;
    }
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);//Af = 0
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);//Af = 0
    std::cout << "A: "<< A<<std::endl;
    std::cout << "A_U: "<<svd.matrixU()<<std::endl;
    std::cout << "A_S: "<<svd.singularValues()<<std::endl;
    std::cout << "A_v: "<<svd.matrixV()<<std::endl;
    cv::Mat cv_A,cv_A_U,cv_A_S,cv_A_V;
    cv::eigen2cv(A,cv_A);

    // cv::SVDecomp(cv_A, cv_A_S, cv_A_U, cv_A_V);
    cv::SVD::compute(cv_A, cv_A_S, cv_A_U, cv_A_V);
    std::cout << "cv_svd_A: "<<cv_A<<std::endl;
    std::cout << "cv_svd_A_U: "<<cv_A_U<<std::endl;
    std::cout << "cv_svd_A_S: "<<cv_A_S<<std::endl;
    std::cout << "cv_svd_A_V: "<<cv_A_V<<std::endl;
    e = (svd.matrixV().block<9,1>(0,8));
    Eigen::Matrix<double,3,3> E;
    std::cout<<"e: "<<e<<std::endl;
    // for(int i = 0 ;i<9;i++)
    // {
    //     E << e(i);
    // }
    E<< e(0,0),e(1,0),e(2,0),e(3,0),e(4,0),e(5,0),e(6,0),e(7,0),e(8,0);
    std::cout<<"E: "<<E<<std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd4constrain(E,Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::cout << "Its singular values are:" << std::endl << svd4constrain.singularValues() << std::endl;
    std::cout << "Its left singular vectors are the columns of the thin U matrix:" <<std::endl << svd4constrain.matrixU() << std::endl;
    std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd4constrain.matrixV() << std::endl;
    double sv = (svd4constrain.singularValues()(0,0) + svd4constrain.singularValues()(1,0))/2;
    Eigen::Matrix<double,3,3> S_E_interm;
    S_E_interm<<  sv,0,0,  0,sv,0  ,0,0,0;
    Eigen::Matrix<double,3,3> E_constrained;
    E_constrained = svd4constrain.matrixU() * S_E_interm * svd4constrain.matrixV().transpose();
    std::cout<<"E_constrained: "<<E_constrained<<std::endl;

    {//recompose origen E verify
        //this is correct form
        Eigen::Matrix<double,3,3> E_recompose, S_E_recompose;
        S_E_recompose<< svd4constrain.singularValues()(0,0),0,0,   0,svd4constrain.singularValues()(1,0),0,  0,0,svd4constrain.singularValues()(2,0);   
        E_recompose = svd4constrain.matrixU() * S_E_recompose * svd4constrain.matrixV().transpose();
        std::cout<<"E_recompose: "<<E_recompose<<std::endl;
    }
    return E_constrained;
    // svd4constrain.matrixU()
    // X = svd.solve(b);
}