#include "pnp.hpp"
inline double getx(const cv::Point2d &p)
{
    return (double)(p.x);
}
inline double gety(const cv::Point2d &p)
{
    return (double)(p.y);
}
inline double getx(const cv::Point3d &p)
{
    return (double)(p.x);
}
inline double gety(const cv::Point3d &p)
{
    return (double)(p.y);
}
inline double getz(const cv::Point3d &p)
{
    return (double)(p.z);
}
Rt_pnp solve_pnp(const Eigen::Matrix<double,3,3> &K, const std::vector<cv::Point2d> &pts_2d,
                 const std::vector<cv::Point3d> &pts_3d)
{
    Rt_pnp res;
    Eigen::Matrix<double,12,12> A;
    int mapi2ptidx[6];
    for(int i = 0; i < 6 ; i++)
    {
        mapi2ptidx[i] = (0 + i*(pts_2d.size()/6)) % pts_2d.size();
    }
    for(int i = 0;i<6; i++)
    {
        cv::Point3d p3d = pts_3d[mapi2ptidx[i]];
        cv::Point2d p2d = pts_2d[mapi2ptidx[i]];
        double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
        Eigen::Matrix<double,1,12> arow1,arow2;
        arow1 << fx * getx(p3d), fx * gety(p3d), fx * getz(p3d), fx,   0,0,0,0,  
                    getx(p3d)*cx-getx(p2d)*getx(p3d), gety(p3d)*cx-getx(p2d)*gety(p3d), getz(p3d)*cx-getx(p2d)*getz(p3d), cx-getx(p2d);
        arow2 << 0,0,0,0,   getx(p3d)*fy, gety(p3d)*fy, getz(p3d)*fy, fy, 
                    getx(p3d)*cy-gety(p2d)*getx(p3d), gety(p3d)*cy-gety(p2d)*gety(p3d), getz(p3d)*cy-gety(p2d)*getz(p3d), cy-gety(p2d);
        A.block<1,12>(i*2,0) = arow1;
        A.block<1,12>(i*2+1,0) = arow2;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::cout << "A: "<< A<<std::endl;
    std::cout << "A_U: "<<svd.matrixU()<<std::endl;
    std::cout << "A_S: "<<svd.singularValues()<<std::endl;
    std::cout << "A_v: "<<svd.matrixV()<<std::endl;
    Eigen::Matrix<double,12,1> p;
    p = svd.matrixV().block<12,1>(0,11);
    Eigen::Matrix<double,3,3> R,R_hat;
    Eigen::Matrix<double,3,1> t,t_hat;
    R_hat << p(0,0),p(1,0),p(2,0),p(4,0),p(5,0),p(6,0),p(8,0),p(9,0),p(10,0);
    Eigen::JacobiSVD<Eigen::MatrixXd> Rsvd(R_hat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = Rsvd.matrixU()*Rsvd.matrixV().transpose();
    double beta = 1.0/(Rsvd.singularValues().mean()); 
    t_hat << p(3,0), p(7,0), p(11,0);
    t = beta * t_hat;

    res.R = R;
    res.t = t;
    return res;

}