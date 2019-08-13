#include "Tslam.hpp"
#define INTERVAL 2
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

void triangulation_(
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

void Tslam::initialize()
{
    char path[50],path1[50];
    sprintf(path, raw_path, image_index);
    Mat img0 = imread(path);
    image_index+=INTERVAL;
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
}
void Tslam::addframe()
{
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
    solvePnP(pts3d_4_pnp,points_4_pnp,K,distortion,R_vec,t_vec,false,SOLVEPNP_ITERATIVE);
    Rodrigues(R_vec,R);
    R = R * (-1);
    t = t_vec *(-1);
    cout<<"t: "<<t<<endl;

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