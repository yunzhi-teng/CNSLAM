#include "triangulation.hpp"
using namespace std;
using namespace cv;
using namespace Eigen;
Matrix<double,3,3> skews(const Matrix<double,3,1>& in)
{
    Matrix<double,3,3> s ;
    s<< 0,-in(2,0),in(1,0),in(2,0),0,-in(0,0),-in(1,0),in(0,0),0;
    return s;

}
Matrix<double, 3,1> linear_triangulation_(const Matrix<double,3,1> &x, const Matrix<double, 3,4> &P, const Matrix<double,3,1> &x1, const Matrix<double,3,4> &P1)
{

    Matrix<double,3,4> Ab1 = skews(x) * P;
    Matrix<double,3,4> Ab2 = skews(x1) * P1;
    Matrix<double,4,3> A;
    A<< Ab1.block<2,3>(0,0), Ab2.block<2,3>(0,0);
    Matrix<double,4,1>  b;
    Matrix<double,3,1> X;
    b << -Ab1.block<2,1>(0,3),-Ab2.block<2,1>(0,3);
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);//A X = b
    X = svd.solve(b);
    return X;

    
}
Matrix<double, 3,1> linear_triangulation(const Matrix<double,3,1> &x, const Matrix<double, 3,4> &P, const Matrix<double,3,1> &x1, const Matrix<double,3,4> &P1)
{
    Matrix<double,3,1> X = linear_triangulation_(x,P,x1,P1);
    return X;
}