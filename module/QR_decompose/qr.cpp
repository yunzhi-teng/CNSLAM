#include <iostream>
#include <fstream>
#include "qr.hpp"
#include  <math.h>
using namespace std;
#define COLS 3
#define ROWS 3
typedef Matrix<double,ROWS,COLS> typemat;
typemat QRDecomposer_base::matrix_R(){
    return R;
}

typemat QRDecomposer_base::matrix_Q(){
    return Q;
}
void QRDecomposer_self::decompose(const typemat &A)
{
    // int rows = A.rows;
    // int cols = A.cols;

    Matrix<double,ROWS,COLS> Q;
    Q.setIdentity(ROWS,COLS);
    Matrix<double,ROWS,COLS> R = A;
    for(int i = 0 ;i < 3;i++)
    {
        Matrix<double,ROWS,COLS> Hi;
        Matrix<double,Dynamic,1> x,x_p,e1, u;//project x to x_p
        Matrix<double,Dynamic,Dynamic> H,I;
        x = R.col(i).tail(COLS-i);
        e1.setZero(COLS-i);
        e1(0,0) = 1;
        x_p = x.norm()*e1;
        if((x_p - x).norm() < 1e-17)
        {
            continue;
        }
        cout<<"e1: "<<e1<<endl;
        u = x - x_p;
        cout<<"u: "<<u<<endl;
        I.setIdentity(ROWS-i,COLS-i);
        H = I - 2 * u * u.transpose()/pow(u.norm(),2);
        Hi.setIdentity(ROWS,COLS);
        Hi.block(i,i,ROWS-i,COLS-i) = H;
        cout<<"H: "<<H<<endl;
        R = Hi * R;
        cout<<"R: "<<R<<endl;
    }
    Q = A*R.inverse();
    cout <<"Q: "<<Q<<endl;
}
void QRDecomposer_eigenHH::decompose(const typemat &A)
{
    HouseholderQR<MatrixXd> qr(A);
    // cout<<"qr_orig: "<<A<<endl;
    qr.compute(A);
    Q = qr.householderQ();
    // cout << "qr_Q: "<<Q<<endl;
    R =  qr.matrixQR().template triangularView<Upper>();
    // cout << "qr_R: "<<R_<<endl;
    // cout<<"veri_qr: "<<Q*R_<<endl;
}
void QRDecomposer_eigenHHcolpiv::decompose(const typemat &A)
{
    ColPivHouseholderQR<MatrixXd> qr(A);
    Q = qr.matrixQ();
    R =  qr.matrixR().template triangularView<Upper>();
}
