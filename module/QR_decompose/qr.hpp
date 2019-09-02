#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
using namespace Eigen;
class QRDecomposer_base {
    public:
        virtual void decompose(const Matrix<double,3,3>& A)= 0;
        Matrix<double,3,3> matrix_R();
        Matrix<double,3,3> matrix_Q();
    protected:
        Matrix<double,3,3> Q,R;
};
class QRDecomposer_self : public QRDecomposer_base {
    public:
        void decompose(const Matrix<double,3,3>& A);
};
class QRDecomposer_eigenHH : public QRDecomposer_base {
    public:
        void decompose(const Matrix<double,3,3>& A);
};
class QRDecomposer_eigenHHcolpiv : public QRDecomposer_base {
    public:
        void decompose(const Matrix<double,3,3>& A);
        // Matrix<double,3,3> matrix_P();
};