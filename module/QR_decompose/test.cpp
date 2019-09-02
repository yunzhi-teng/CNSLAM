#include "qr.hpp"
#include "iostream"
using namespace std;
void factorizeKRT_wrapper(const Matrix<double,3,4>& p)
{
    Matrix<double,3,3> A = p.block(0,0,3,3).inverse();
    // A << 1,2,1,2,3,2,1,2,3;
    cout<<"mat decomposed : "<<A<<endl;

    QRDecomposer_base *self, *eigen_HH, *eigen_HHcolpiv;
    self = new QRDecomposer_self();
    eigen_HH = new QRDecomposer_eigenHH();
    eigen_HHcolpiv = new QRDecomposer_eigenHHcolpiv();
    self->decompose(A);
    eigen_HH->decompose(A);
    eigen_HHcolpiv->decompose(A);

}
int main()
{
    //projection matrix
    Matrix<double,3,4> Po1;
    Po1<< 976.5, 53.82, -239.8, 3.875e+05, 98.49, 933.3, 157.4, 2.428e+05, 0.5790, 0.1108, 0.8077, 1118;
    cout<<"po1:" << Po1<<endl;
    factorizeKRT_wrapper(Po1);

}