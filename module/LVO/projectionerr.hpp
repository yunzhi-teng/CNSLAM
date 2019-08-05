#include <stdio.h>
#include <iostream>
struct TReprojectionError
{ //calibrated camera

    TReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}
    template <typename T>
    bool operator()(
                    const T *const camera_pose,
                    const T *const point,         //world coordinate
                    const T *const camera_intrin,//fx,fy,cx,cy
                    T *residuals
                    ) const
    {
        // camera_pose[0,1,2] are the angle-axis rotation.
        T p[3]; //camera coordinate
        // printf("\n3dpoint: ");
        // for(int i = 0; i<3; i++)
        // {
        //     // printf("%f,",(point[i]));
        //     std::cout<<point[i]<<",";

        // }
        // printf("\n");
        // printf("\ncamerapose3: ");
        // for(int i = 0; i<3; i++)
        // {
        //     printf("%f,",(camera_pose[i]));

        // }
        // printf("\n");
        ceres::AngleAxisRotatePoint(camera_pose, point, p);
        // camera_pose[3,4,5] are the translation.
        p[0] += camera_pose[3];
        p[1] += camera_pose[4];
        p[2] += camera_pose[5];
        // T u = fx * x / z + cx
        // printf("\n3dpoint after Rt: ");
        // for(int i = 0; i<3; i++)
        // {
        //     printf("%f,",(p[i]));

        // }
        // printf("\n");
        T predicted_x, predicted_y;
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        predicted_x = camera_intrin[0] *xp  + camera_intrin[1];
        predicted_y = camera_intrin[2] *yp + camera_intrin[3];

        // Compute final projected point position.

        // The error is the difference between the predicted and observed position.

        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        // std::cout<<"residual: x:"<<residuals[0]<<", y:"<<residuals[1]<<std::endl;
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<TReprojectionError, 2, 6, 3, 4>(
            new TReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;

};