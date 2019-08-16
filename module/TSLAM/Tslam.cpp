#include "Tslam.hpp"

char *raw_path = "/data/SLAM/dataset/kitti/00/image_0/%06d.png";


Point2f pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2f(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}


void triangulation(
    const vector<KeyPoint> &keypoint_0,//reference,last frame
    const vector<KeyPoint> &keypoint_1,//current frame
    const std::vector<DMatch> &matches,//only 3dpoint never meet in match
    const Mat &R0, const Mat &t0,
    const Mat &R1, const Mat &t1,
    const int kf_idx0, const int kf_idx1,
    const Mat& K,
    vector<Mappoint> &mappoint)
{
    Mat T1 = (Mat_<float>(3, 4) << R0.at<double>(0, 0), R0.at<double>(0, 1), R0.at<double>(0, 2), t0.at<double>(0, 0),
              R0.at<double>(1, 0), R0.at<double>(1, 1), R0.at<double>(1, 2), t0.at<double>(1, 0),
              R0.at<double>(2, 0), R0.at<double>(2, 1), R0.at<double>(2, 2), t0.at<double>(2, 0));
    Mat T2 = (Mat_<float>(3, 4) << R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0, 0),
              R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1, 0),
              R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2, 0));
    vector<Point2f> pts_1, pts_2;
    
    for (DMatch m : matches)
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back(pixel2cam(keypoint_0[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_1[m.trainIdx].pt, K));
        Mappoint mp;
        mp.keyframe_indexed_pt2d.push_back(m.queryIdx);
        mp.keyframe_indexed_pt2d.push_back(m.trainIdx);
        mp.visible_keyframe_idx.push_back(kf_idx0);
        mp.visible_keyframe_idx.push_back(kf_idx1);
        mappoint.push_back(mp);
    }
    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for (int i = 0; i < pts_4d.cols; i++)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 归一化
        mappoint[i].pt3d.x = x.at<float>(0, 0);
        mappoint[i].pt3d.y = x.at<float>(1, 0);
        mappoint[i].pt3d.z = x.at<float>(2, 0);
    }
}


void Tslam::initialize()
{
    char path[50],path1[50];
    sprintf(path, raw_path, image_index);
    Mat img0 = imread(path);
    image_index+=INITIAL_INTER;
    sprintf(path1, raw_path, image_index);
    Mat img1 = imread(path1);
    image_index+=INTERVAL;

    Frameproccessing frame0,frame1;
    
    imshow("image",img1);
    waitKey(1);
    vector<DMatch> matches;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    detector->detect(img0, frame0.keyframe_.keypoint);
    detector->detect(img1, frame1.keyframe_.keypoint);

    descriptor->compute(img0, frame0.keyframe_.keypoint, frame0.keyframe_.desc);
    descriptor->compute(img1, frame1.keyframe_.keypoint, frame1.keyframe_.desc);

    vector<DMatch> match;

    matcher->match(frame0.keyframe_.desc, frame1.keyframe_.desc, match);
    double min_dist = 8888, max_dist = 0;
    for (int i = 0; i < frame0.keyframe_.desc.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist < max_dist)
            max_dist = dist;
    }
    for (int i = 0; i < frame0.keyframe_.desc.rows; i++)
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
    // cout<<"total matches:"<<matches.size()<<endl;
    Mat R, t;
    vector<Point2f> points1, points2;
    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(frame0.keyframe_.keypoint[matches[i].queryIdx].pt);
        points2.push_back(frame1.keyframe_.keypoint[matches[i].trainIdx].pt);
    }

    Point2d principal_point(6.071928000000e+02, 1.852157000000e+02);
    double focal_length = 7.188560000000e+02;
    Mat essential_mat;
    Mat confi;
    essential_mat = findEssentialMat(points1, points2, focal_length, principal_point, 8, 0.99999, 3, confi);
    recoverPose(essential_mat, points1, points2, R, t, focal_length, principal_point);

    //keyframe
    frame0.keyframe_.R = Mat::zeros(3, 3, CV_64F);
    frame0.keyframe_.R.at<double>(0) = 1;
    frame0.keyframe_.R.at<double>(4) = 1;
    frame0.keyframe_.R.at<double>(8) = 1;
    frame0.keyframe_.t = Mat::zeros(3, 1, CV_64F);
    frame1.keyframe_.R = R;
    frame1.keyframe_.t = t;
    int keyframe_index0,keyframe_index1;
    all_keyframe.push_back(frame0.keyframe_);
    keyframe_index0 = all_keyframe.size()-1;
    all_keyframe.push_back(frame1.keyframe_);
    keyframe_index1 = all_keyframe.size()-1;

    //mappoint
    vector<Mappoint> mappoint_temp;
    triangulation(frame0.keyframe_.keypoint,frame1.keyframe_.keypoint,matches,frame0.keyframe_.R,frame0.keyframe_.t,frame1.keyframe_.R,frame1.keyframe_.t,keyframe_index0,keyframe_index1,K,mappoint_temp);

    all_mappoint.insert(all_mappoint.end(), mappoint_temp.begin(),mappoint_temp.end());

    //frame_processing
    for(int i = 0; i< frame1.keyframe_.keypoint.size();i++)//same size as keypoint, matchkp_mp[kp_idx] =mp_idx, all_mappoint[mp_idx]
    {
        int mp_idx = -1;
        for(int j = 0; j<all_mappoint.size();j++)
        {
            for(int k = 0; k <all_mappoint[j].visible_keyframe_idx.size();k++)
            {
                if(keyframe_index1 ==all_mappoint[j].visible_keyframe_idx[k])
                {
                    if(all_mappoint[j].keyframe_indexed_pt2d[k] == i)
                    {
                        mp_idx = j;
                    }
                }
            }
        }
        // for(int j = 0; j< all_mappoint.size();j++)
        // {
        //     if(kf_idx == all_mappoint[j].visible_keyframe_idx[visible_keyframe_idx.size()-1])
        //     {
        //         if(all_mappoint[j].keyframe_indexed_pt2d[k] == i)
        //         {
        //             mp_idx = j;
        //         }
        //     }
        // }
        frame1.matchkp_mp.push_back(mp_idx);
    }
    last_frame = frame1;
    // ba();
}
void opt_campose_2Rt(const Eigen::Matrix<double, 3, 2>& cam_pose,Mat& R,Mat& t)
{
        Eigen::Matrix<double,3,1> R_vec_opt;
        R_vec_opt(0,0) = cam_pose(0,0);
        R_vec_opt(1,0) = cam_pose(1,0);
        R_vec_opt(2,0) = cam_pose(2,0);
        Eigen::Matrix<double, 3, 3> R_opt_e;
        ceres::AngleAxisToRotationMatrix<double>(R_vec_opt.data(),R_opt_e.data());
        Mat Ropt;
        eigen2cv(R_opt_e,Ropt);
        R = Ropt;
        t.at<double>(0) = cam_pose(0, 1);
        t.at<double>(1) = cam_pose(1, 1);
        t.at<double>(2) = cam_pose(2, 1);
}

void Tslam::addframe()
{
    // if(image_index %5 ==0)
    {
        ba();
    }
    char path[50];
    sprintf(path, raw_path, image_index);
    Mat img0 = imread(path);
    image_index+=INTERVAL;
    imshow("image",img0);
    waitKey(1);
    Frameproccessing frame_;
    vector<DMatch> matches;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    detector->detect(img0, frame_.keyframe_.keypoint);
    descriptor->compute(img0, frame_.keyframe_.keypoint, frame_.keyframe_.desc);
    vector<DMatch> match;
    matcher->match(last_frame.keyframe_.desc, frame_.keyframe_.desc, match);

    double min_dist = 8888, max_dist = 0;
    for (int i = 0; i < last_frame.keyframe_.desc.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist < max_dist)
            max_dist = dist;
    }
    for (int i = 0; i < last_frame.keyframe_.desc.rows; i++)
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
    // cout<<"total matches:"<<matches.size()<<endl;
    vector<DMatch> matches_4_pnp;
    vector<DMatch> matches_4_triang;
    for(int i = 0; i< matches.size(); i++)
    {
        if(last_frame.matchkp_mp[matches[i].queryIdx] != -1)
        {
            matches_4_pnp.push_back(matches[i]);
        }
        else
        {
            matches_4_triang.push_back(matches[i]);
        }
        
    }

    Mat R, t, R_vec, t_vec, distortion;
    vector<Point2f> points_4_pnp;
    vector<Point3f> pts3d_4_pnp;
    for (int i = 0; i < (int)matches_4_pnp.size(); i++)
    {
        pts3d_4_pnp.push_back(all_mappoint[last_frame.matchkp_mp[matches_4_pnp[i].queryIdx]].pt3d);
        points_4_pnp.push_back(frame_.keyframe_.keypoint[matches_4_pnp[i].trainIdx].pt);
    }
    cout<<"pts3d_size: "<<pts3d_4_pnp.size()<<endl;
    // cout<<"pnp_pts3d: "<<pts3d_4_pnp<<endl;
    // cout<<"pnp_pts2d: "<<points_4_pnp<<endl;
    solvePnP(pts3d_4_pnp,points_4_pnp,K,distortion,R_vec,t_vec,false,SOLVEPNP_ITERATIVE);

    Rodrigues(R_vec,R);
    R = R * (1);
    t = t_vec *(1);
    cout<<"t: "<<t<<endl;

// {//perframe ba
//         Eigen::Matrix<double, 3, 3> R_eigen;
//         Eigen::Matrix<double, 3, 1> axis_angle_eigen;
//         cv2eigen(R, R_eigen);
//         ceres::RotationMatrixToAngleAxis<double>(R_eigen.data(), axis_angle_eigen.data());

//         Eigen::Matrix<double, 3, 2> camera_pose_;
//         for (int i = 0; i < 3; i++)
//         {
//             ((double *)(camera_pose_.data()))[i] = axis_angle_eigen(i, 0);
//         }
//         for (int i = 0; i < 3; i++)
//         {
//             ((double *)(camera_pose_.data()))[i + 3] = t.at<double>(i);
//         }

//         double camera_intrin_[4];
//         camera_intrin_[0] = K.at<double>(0);
//         camera_intrin_[1] = K.at<double>(2);
//         camera_intrin_[2] = K.at<double>(4);
//         camera_intrin_[3] = K.at<double>(5);
//         ceres::Problem::Options problem_options;
//         ceres::Problem problem(problem_options);
//         for (int i = 0; i < points_4_pnp.size(); i++)
//         {
//             problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TReprojectionError, 2, 6, 1,1,1, 4>(
//                                          new TReprojectionError(points_4_pnp[i].x, points_4_pnp[i].y)),
//                                      NULL,
//                                      camera_pose_.data(),
//                                      &(all_mappoint[last_frame.matchkp_mp[matches_4_pnp[i].queryIdx]].pt3d.x),
//                                      &(all_mappoint[last_frame.matchkp_mp[matches_4_pnp[i].queryIdx]].pt3d.y),
//                                      &(all_mappoint[last_frame.matchkp_mp[matches_4_pnp[i].queryIdx]].pt3d.z),
//                                      camera_intrin_);
//             problem.SetParameterBlockConstant(camera_intrin_);
//             vector<double*> parameter_block;
//             parameter_block.push_back(camera_pose_.data());
//             parameter_block.push_back(&(all_mappoint[last_frame.matchkp_mp[matches_4_pnp[i].queryIdx]].pt3d.x));
//             parameter_block.push_back(&(all_mappoint[last_frame.matchkp_mp[matches_4_pnp[i].queryIdx]].pt3d.y));
//             parameter_block.push_back(&(all_mappoint[last_frame.matchkp_mp[matches_4_pnp[i].queryIdx]].pt3d.z));
//             parameter_block.push_back(camera_intrin_);
//             auto evaluatoption = ceres::Problem::EvaluateOptions();
//             evaluatoption.parameter_blocks = parameter_block;
//             vector<double> residules;
//             double cost;
//             bool fa =  problem.Evaluate(ceres::Problem::EvaluateOptions(),&cost,&residules,NULL ,NULL);
//             cout<<"cost:"<<cost<<endl;
//                 //     cout<<"residuals:"<<residules[0]<<", "<<residules[1]<<endl;
//         //     problem.ceres::CostFunction::Evaluate();
//         }
//         ceres::Solver::Options options;
//         options.use_nonmonotonic_steps = true;
//         options.preconditioner_type = ceres::SCHUR_JACOBI;
//         options.linear_solver_type = ceres::ITERATIVE_SCHUR;
//         options.use_inner_iterations = true;
//         options.max_num_iterations = 40;
//         options.minimizer_progress_to_stdout = true;

//         // // // solve
//         ceres::Solver::Summary summary;
//         ceres::Solve(options, &problem, &summary);
//         // cout << summary.BriefReport() << endl;
//         opt_campose_2Rt(camera_pose_,R,t);

// }
    //keyframe
    frame_.keyframe_.R = R;
    frame_.keyframe_.t = t;
    all_keyframe.push_back(frame_.keyframe_);
    int kf_idx = all_keyframe.size()-1;

    //mappoint
    for (int i = 0; i < (int)matches_4_pnp.size(); i++)
    {
        all_mappoint[last_frame.matchkp_mp[matches_4_pnp[i].queryIdx]].visible_keyframe_idx.push_back(kf_idx);
        all_mappoint[last_frame.matchkp_mp[matches_4_pnp[i].queryIdx]].keyframe_indexed_pt2d.push_back(matches_4_pnp[i].trainIdx);
    }


    vector<Mappoint> mappoint_temp;
    triangulation(last_frame.keyframe_.keypoint,frame_.keyframe_.keypoint,matches_4_triang,last_frame.keyframe_.R,last_frame.keyframe_.t,R,t,kf_idx-1,kf_idx,K,mappoint_temp);
    all_mappoint.insert(all_mappoint.end(), mappoint_temp.begin(),mappoint_temp.end());
    
    //frame
    for(int i = 0; i< frame_.keyframe_.keypoint.size();i++)
    {
        int mp_idx = -1;
        for(int j = 0; j< all_mappoint.size();j++)
        {
            for( int k = 0; k < all_mappoint[j].visible_keyframe_idx.size();k ++)
            {
                if(kf_idx == all_mappoint[j].visible_keyframe_idx[k])
                {
                    if(all_mappoint[j].keyframe_indexed_pt2d[k] == i)
                    {
                        mp_idx = j;
                    }
                }
            }
        }

        // for(int j = 0; j< all_mappoint.size();j++)
        // {
        //     if(kf_idx == all_mappoint[j].visible_keyframe_idx[visible_keyframe_idx.size()-1])
        //     {
        //         if(all_mappoint[j].keyframe_indexed_pt2d[k] == i)
        //         {
        //             mp_idx = j;
        //         }
        //     }
        // }
        frame_.matchkp_mp.push_back(mp_idx);
    }
    last_frame = frame_;

}
void Tslam::ba()
{

    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    //intrin
    double camera_intrin_[4];
    camera_intrin_[0] = K.at<double>(0);
    camera_intrin_[1] = K.at<double>(2);
    camera_intrin_[2] = K.at<double>(4);
    camera_intrin_[3] = K.at<double>(5);
    //pose vector
    vector<Eigen::Matrix<double, 3, 2>> cam_poses;
    for(int i = 0;i< all_keyframe.size();i++)
    {
        Eigen::Matrix<double, 3, 3> R_eigen;
        Eigen::Matrix<double, 3, 1> axis_angle_eigen;
        // cv2eigen(all_keyframe[i].R.inv(), R_eigen);
         cv2eigen(all_keyframe[i].R, R_eigen);
        ceres::RotationMatrixToAngleAxis<double>(R_eigen.data(), axis_angle_eigen.data());
        
        Eigen::Matrix<double, 3, 2> camera_pose_;
        for (int j = 0; j < 3; j++)
        {
            ((double *)(camera_pose_.data()))[j] = axis_angle_eigen(j, 0);
        }
        for (int j = 0; j < 3; j++)
        {
            // Mat t_ = (-1)* all_keyframe[i].R.inv()*all_keyframe[i].t;
            ((double *)(camera_pose_.data()))[j + 3] = all_keyframe[i].t.at<double>(j);
        }
        cam_poses.push_back(camera_pose_);
    }

//opt one keyframe
    static int iii= all_keyframe.size()-1;
    if(iii<all_keyframe.size())
    {
        for(int i = 0;i<all_mappoint.size();i++)
        {
            for(int j = 0; j<all_mappoint[i].visible_keyframe_idx.size();j++)
            {
                if(all_mappoint[i].visible_keyframe_idx[j] == iii)
                {
                    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TReprojectionError, 2, 6, 1,1,1, 4>(
                                                new TReprojectionError(all_keyframe[iii].keypoint[all_mappoint[i].keyframe_indexed_pt2d[j]].pt.x, all_keyframe[iii].keypoint[all_mappoint[i].keyframe_indexed_pt2d[j]].pt.y)),
                                            NULL,
                                            cam_poses[iii].data(),
                                            //  t3dpoint+i*3,
                                            &(all_mappoint[i].pt3d.x),
                                            &(all_mappoint[i].pt3d.y),
                                            &(all_mappoint[i].pt3d.z),
                                            camera_intrin_);
                    problem.SetParameterBlockConstant(camera_intrin_);
                    vector<double*> parameter_block;
                    parameter_block.push_back(cam_poses[iii].data());
                    parameter_block.push_back(&(all_mappoint[i].pt3d.x));
                    parameter_block.push_back(&(all_mappoint[i].pt3d.y));
                    parameter_block.push_back(&(all_mappoint[i].pt3d.z));
                    parameter_block.push_back(camera_intrin_);
                    auto evaluatoption = ceres::Problem::EvaluateOptions();
                    evaluatoption.parameter_blocks = parameter_block;
                    vector<double> residules;
                    double cost;
                    bool fa =  problem.Evaluate(ceres::Problem::EvaluateOptions(),&cost,&residules,NULL ,NULL);
                    cout<<"cost:"<<cost<<endl;
                }
            }
        }



        ceres::Solver::Options options;
        options.use_nonmonotonic_steps = true;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.use_inner_iterations = true;
        options.max_num_iterations = 80;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        cout << summary.BriefReport() << endl;

        Eigen::Matrix<double,3,1> R_vec_opt;
        R_vec_opt(0,0) = cam_poses[iii](0,0);
        R_vec_opt(1,0) = cam_poses[iii](1,0);
        R_vec_opt(2,0) = cam_poses[iii](2,0);

        Eigen::Matrix<double, 3, 3> R_opt_e;
        ceres::AngleAxisToRotationMatrix<double>(R_vec_opt.data(),R_opt_e.data());
        Mat Ropt;
        eigen2cv(R_opt_e,Ropt);
        all_keyframe[iii].R = Ropt;
        all_keyframe[iii].t.at<double>(0) = cam_poses[iii](0, 1);
        all_keyframe[iii].t.at<double>(1) = cam_poses[iii](1, 1);
        all_keyframe[iii].t.at<double>(2) = cam_poses[iii](2, 1);
        last_frame.keyframe_.R = all_keyframe[all_keyframe.size()-1].R;
        last_frame.keyframe_.t = all_keyframe[all_keyframe.size()-1].t;
        iii++;
    }
    // if(image_index % 3 ==0)
    // {
    //     for(int i =0;i<all_mappoint.size();i++)//opt all keyframe
    //     {
    //         // cout<<"3dpt: "<<all_mappoint[i].pt3d<<endl;
    //         for(int j = 0; j< all_mappoint[i].visible_keyframe_idx.size();j++)
    //         {
    //             //all_keyframe[all_mappoint[i].visible_keyframe_idx[j]]
    //             // cout<<"2dpt: "<<all_keyframe[all_mappoint[i].visible_keyframe_idx[j]].keypoint[all_mappoint[i].keyframe_indexed_pt2d[j]].pt<<endl;
    //             // if(all_mappoint[i].visible_keyframe_idx[j] == 3)
    //             // {
    //                 problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TReprojectionError, 2, 6, 1,1,1,4>(
    //                                             new TReprojectionError(all_keyframe[all_mappoint[i].visible_keyframe_idx[j]].keypoint[all_mappoint[i].keyframe_indexed_pt2d[j]].pt.x, all_keyframe[all_mappoint[i].visible_keyframe_idx[j]].keypoint[all_mappoint[i].keyframe_indexed_pt2d[j]].pt.y)),
    //                                         NULL,
    //                                         cam_poses[all_mappoint[i].visible_keyframe_idx[j]].data(),
    //                                         //  t3dpoint+i*3,
    //                                         &(all_mappoint[i].pt3d.x),
    //                                         &(all_mappoint[i].pt3d.y),
    //                                         &(all_mappoint[i].pt3d.z),
    //                                         camera_intrin_);
    //                 problem.SetParameterBlockConstant(camera_intrin_);
    //             // }

    //         }
    //     }
    //     ceres::Solver::Options options;
    //     options.use_nonmonotonic_steps = true;
    //     options.preconditioner_type = ceres::SCHUR_JACOBI;
    //     options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    //     options.use_inner_iterations = true;
    //     options.max_num_iterations = 20;
    //     options.minimizer_progress_to_stdout = true;
    //     ceres::Solver::Summary summary;
    //     ceres::Solve(options, &problem, &summary);
    //     cout << summary.BriefReport() << endl;

    //     for(int i = 0;i<all_keyframe.size();i++)
    //     {
    //         Eigen::Matrix<double,3,1> R_vec_opt;
    //         R_vec_opt(0,0) = cam_poses[i](0,0);
    //         R_vec_opt(1,0) = cam_poses[i](1,0);
    //         R_vec_opt(2,0) = cam_poses[i](2,0);

    //         Eigen::Matrix<double, 3, 3> R_opt_e;
    //         ceres::AngleAxisToRotationMatrix<double>(R_vec_opt.data(),R_opt_e.data());
    //         Mat Ropt;
    //         eigen2cv(R_opt_e,Ropt);
    //         all_keyframe[i].R = Ropt;
    //         all_keyframe[i].t.at<double>(0) = cam_poses[i](0, 1);
    //         all_keyframe[i].t.at<double>(1) = cam_poses[i](1, 1);
    //         all_keyframe[i].t.at<double>(2) = cam_poses[i](2, 1);

    //     }
    //     last_frame.keyframe_.R = all_keyframe[all_keyframe.size()-1].R;
    //     last_frame.keyframe_.t = all_keyframe[all_keyframe.size()-1].t;
    // }
}