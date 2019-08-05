#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/opencv.hpp"
#include <fstream>
#include "projectionerr.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace cv;
using namespace std;
char *raw_path = "/data/SLAM/dataset/kitti/00/image_0/%06d.png";
Mat getimage(int i)
{
    char path[50];
    sprintf(path, raw_path, i);
    Mat img = imread(path);
    return img;
}
Mat VO(int image_index, Mat &x, Mat &Rinv_accu);
std::vector<vector<float>> get_Pose(const std::string &path)
{

    std::vector<vector<float>> poses;
    ifstream myfile(path);
    string line;
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            char *dup = strdup(line.c_str());
            char *token = strtok(dup, " ");
            std::vector<float> v;
            while (token != NULL)
            {
                v.push_back(atof(token));
                token = strtok(NULL, " ");
            }
            poses.push_back(v);
            free(dup);
        }
        myfile.close();
    }
    else
    {
        cout << "Unable to open file";
    }

    return poses;
}

int main()
{
    int image_index = 0;
    Mat traj = Mat::zeros(600, 600, CV_8UC3);
    vector<vector<float>> poses = get_Pose("/data/SLAM/dataset/poses/00.txt");
    Mat x = Mat::zeros(3, 1, CV_64F);
    Mat Rinv_accumulate = Mat::zeros(3, 3, CV_64F);
    Rinv_accumulate.at<double>(0) = 1;
    Rinv_accumulate.at<double>(4) = 1;
    Rinv_accumulate.at<double>(8) = 1;
    // Mat R = Mat::zeros(3,3,CV_64F);
    for (int i = 0; i < 1000; i++)
    {
        Mat tt = VO(image_index, x, Rinv_accumulate);
        image_index++;

        Point2f center = Point2f(-(int(x.at<double>(0, 0))) + 300, -(int(x.at<double>(2, 0))) + 100);

        circle(traj, center, 1, Scalar(0, 0, 255), 2);

        Point2f t_center = Point2f(int(poses[i][3]) + 300, int(poses[i][11]) + 100);
        circle(traj, t_center, 1, Scalar(255, 0, 0), 2);

        // cout << t.t() << endl;
        // cout << "["<<poses[i][3] << ", " << poses[i][7] << ", " << poses[i][11] <<"]"<<endl;
        // cout<<-(t.at<float>(2))<<endl;
        // cout<<"pose:"<<poses[i][11]<<endl;
        imshow("Trajectory", traj);
        
        waitKey(1);
    }
}
Point2f pixel2cam(const Point2d &p, const Mat &K);
void triangulation(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    // const Mat &R_prev, const Mat &t_prev,
    vector<Point3d> &points);

Mat VO(int image_index, Mat &x, Mat &Rinv_accu)
{
    static Mat R_prev, tr_prev, x_prev;
    static bool first_in = true;
    Mat desc1, desc2;
    Mat img1 = getimage(image_index);
    Mat img2 = getimage(image_index + 1);
    
    imshow("image",img2);
    waitKey(1);
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    descriptor->compute(img1, keypoints1, desc1);
    descriptor->compute(img2, keypoints2, desc2);

    vector<DMatch> match;

    matcher->match(desc1, desc2, match);
    double min_dist = 8888, max_dist = 0;
    for (int i = 0; i < desc1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist < max_dist)
            max_dist = dist;
    }
    for (int i = 0; i < desc1.rows; i++)
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
    // cout<<"total matches:"<<matches.size()<<endl;
    Mat R, t;
    Mat K = (Mat_<double>(3, 3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);

    vector<Point2f> points1, points2;
    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    Point2d principal_point(6.071928000000e+02, 1.852157000000e+02);
    double focal_length = 7.188560000000e+02;
    Mat essential_mat;
    Mat confi;
    essential_mat = findEssentialMat(points1, points2, focal_length, principal_point, 8, 0.99999, 3, confi);

    recoverPose(essential_mat, points1, points2, R, t, focal_length, principal_point);
    // cout << "t:" << t.t() << endl;
    // cout << "R: " << R << endl;
    double s = sum(confi)[0];
    // if (s / confi.rows / confi.cols > 0.99)
    // {

    // x = R * x + t;
    if (first_in == true)
    {
        first_in = false;
        x =  R.inv()*((-1)*t);//to c1 == world coord
        Rinv_accu = R.inv();
    }
    else
    {
        
        
        // }
        // else
        // {
        //     x = R_prev * x + t_prev;
        //     R_accu*=R_prev;
        // }
        // cout<<"to:"<<x.t()<<endl;
        // cout<<"to2: "<<(x.at<double>(2,0)) <<endl;
        /**testing eigen&ceres***/
        Eigen::Matrix<double, 3, 3> R_eigen;
        Eigen::Matrix<double, 3, 1> axis_angle_eigen;
        cv2eigen(R, R_eigen);
        ceres::RotationMatrixToAngleAxis<double>(R_eigen.data(), axis_angle_eigen.data());
        // cout << "R_eigen:" << R_eigen << endl;
        // cout << "axie_agle_eigen:" << axis_angle_eigen << endl;
        // Eigen::Matrix<double, 3, 3> R_eigen_verifying;
        // ceres::AngleAxisToRotationMatrix<double>(axis_angle_eigen.data(), R_eigen_verifying.data());
        // cout << "R_ei_veryfying:" << R_eigen_verifying << endl;
        // /******/


        vector<Point3d> point_3D;
        triangulation(keypoints1, keypoints2, matches, R, t, point_3D);

        Eigen::Matrix<double, 3, 2> camera_pose_;
        for (int i = 0; i < 3; i++)
        {
            ((double *)(camera_pose_.data()))[i] = axis_angle_eigen(i, 0);
        }
        for (int i = 0; i < 3; i++)
        {
            ((double *)(camera_pose_.data()))[i + 3] = t.at<double>(i);
        }

        // //  << , axis_angle_eigen(1, 0), axis_angle_eigen(2, 0), t.at<double>(0), t.at<double>(1), t.at<double>(2);
        double camera_intrin_[4];
        camera_intrin_[0] = K.at<double>(0);
        camera_intrin_[1] = K.at<double>(2);
        camera_intrin_[2] = K.at<double>(4);
        camera_intrin_[3] = K.at<double>(5);
        // cout << "cam_pose_:" << camera_pose_ << endl;
        // cout << " cam intrin:" << camera_intrin_[0] << "," << camera_intrin_[1] << "," << camera_intrin_[2] << "," << camera_intrin_[3] << "\n";
        ceres::Problem::Options problem_options;
        ceres::Problem problem(problem_options);
        double *t3dpoint = (double *)malloc(sizeof(double) * point_3D.size() * 3);
        double *ptr3d = t3dpoint;
        for (int i = 0; i < point_3D.size(); i += 1)
        {
            *ptr3d = point_3D[i].x;
            *(ptr3d + 1) = point_3D[i].y;
            *(ptr3d + 2) = point_3D[i].z;
            ptr3d += 3;
        }
        // printf("\n###");
        // for (int i = 0; i < point_3D.size(); i++)
        // {
        //     printf("%d,", t3dpoint[i] / 500);
        // }

        for (int i = 0; i < point_3D.size(); i++)
        {
        //     // printf("\naddresidula3dpoint: ");
        //     // for(int j = 0; j<3; j++)
        //     // {
        //     //     printf("%f,",((t3dpoint + i*3)[j]));

        //     // }
        //     // printf("\n");
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TReprojectionError, 2, 6, 3, 4>(
                                         new TReprojectionError(points2[i].x, points2[i].y)),
                                     NULL,
                                     camera_pose_.data(),
                                     t3dpoint + i * 3,
                                     camera_intrin_);
            problem.SetParameterBlockConstant(camera_intrin_);

            // vector<double*> parameter_block;
            // parameter_block.push_back(camera_pose_.data());
            // parameter_block.push_back(t3dpoint + i * 3);
            // parameter_block.push_back(camera_intrin_);
            // auto evaluatoption = ceres::Problem::EvaluateOptions();
            // evaluatoption.parameter_blocks = parameter_block;
            // cout<<"point"<<*(double*)(t3dpoint + i * 3)<<endl;
            // vector<double> residules;
            // double cost;
            // bool fa =  problem.Evaluate(ceres::Problem::EvaluateOptions(),&cost,&residules,NULL ,NULL);
            // cout<<"cost:"<<cost<<endl;
            // CHECK(fa);
            // cout<<"residuals:"<<residules[0]<<", "<<residules[1]<<endl;
            // problem.ceres::CostFunction::Evaluate();
            // if(   //residules[0]*residules[0] +residules[1]*residules[1] > 19999
            //   !fa || cost > 1e+07)
            // {
                // problem.SetParameterBlockConstant(t3dpoint + i*3);
                // problem.SetParameterBlockConstant(camera_pose_.data());
            //     cout<<"froze"<<endl;

            // }
        }
        // // configure the solver
        ceres::Solver::Options options;
        options.use_nonmonotonic_steps = true;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.use_inner_iterations = true;
        options.max_num_iterations = 40;
        options.minimizer_progress_to_stdout = true;

        // // // solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);


        axis_angle_eigen(0, 0) = camera_pose_(0, 0);
        axis_angle_eigen(1, 0) = camera_pose_(1, 0);
        axis_angle_eigen(2, 0) = camera_pose_(2, 0);
        t.at<double>(0) = camera_pose_(0, 1);
        t.at<double>(1) = camera_pose_(1, 1);
        t.at<double>(2) = camera_pose_(2, 1);

        Eigen::Matrix<double, 3, 3> R_afteropt_verifying;
        ceres::AngleAxisToRotationMatrix<double>(axis_angle_eigen.data(), R_afteropt_verifying.data());
        // cout << "R_afteropt_veryfying:" << R_afteropt_verifying << endl;
        cout << summary.BriefReport() << endl;
        Mat Ropt;
        eigen2cv(R_afteropt_verifying,Ropt);
        // R_accu = Ropt;

        // printf("final\n");
        R = Ropt;

        x = Rinv_accu*R.inv() *((-1)*t) + x - 2* tr_prev;
        cout<<"x:"<<x<<endl;
        free(t3dpoint);
    }
    tr_prev = Rinv_accu *R.inv()*((-1)*t);
    Rinv_accu = Rinv_accu * R.inv(); 
    // x_prev = x;
    return t;
}

Point2f pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2f(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void triangulation(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points)
{
    Mat T1 = (Mat_<float>(3, 4) << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    Mat K = (Mat_<double>(3, 3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);
    vector<Point2f> pts_1, pts_2;
    for (DMatch m : matches)
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // 转换成非齐次坐标
    for (int i = 0; i < pts_4d.cols; i++)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 归一化
        Point3d p(
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0));
        points.push_back(p);
    }
}
