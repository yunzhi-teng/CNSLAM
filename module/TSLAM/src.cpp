#include "Tslam.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
using namespace cv;
using namespace std;

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
void save_pcl(const int width,const int height,const vector<Mappoint>& mappoint,const string path)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width = width;
    cloud.height = height;
    cloud.is_dense = false;
    cloud.points.resize(cloud.width *cloud.height);
    int i = 0;
    for(; i< mappoint.size();i++)
    {
        cloud.points[i].x = mappoint[i].pt3d.x;
        cloud.points[i].y = mappoint[i].pt3d.y;
        cloud.points[i].z = mappoint[i].pt3d.z;

    }
    for(;i<cloud.points.size();i++)
    {
        cloud.points[i].x =0;
        cloud.points[i].y =0;
        cloud.points[i].z =0;
    }
    pcl::io::savePCDFileASCII(path,cloud);
}
int main()
{
    Tslam tslam;
    tslam.initialize();
    save_pcl(100,100,tslam.all_mappoint,"init.pcd");
        cout<<"map point_init: "<<endl;
        for(int j = 0; j < tslam.all_mappoint.size();j++)
        {
            cout<<tslam.all_mappoint[j].pt3d<<", ";

        }
        cout<<endl;
    Mat traj = Mat::zeros(600, 600, CV_8UC3);
    vector<vector<float>> poses = get_Pose("/data/SLAM/dataset/poses/00.txt");

    for (int i = 0; i < 11; i++)
    {
        cout<<"frame: "<<tslam.image_index<<endl;
        tslam.addframe();
        Mat x = tslam.last_frame.keyframe_.t;
        Point2f center = Point2f(-(int(x.at<double>(0, 0))) + 300, -(int(2*x.at<double>(2, 0))) + 100);
        cout<<"pred: "<<int(x.at<double>(0, 0))<<"; "<<int(2*x.at<double>(2, 0))<<endl;
        circle(traj, center, 1, Scalar(0, 0, 255), 2);

        Point2f t_center = Point2f(int(poses[i*INTERVAL+INITIAL_INTER][3]) + 300, int(poses[i*INTERVAL+INITIAL_INTER][11]) + 100);
        cout<<"gt: "<<int(poses[i*INTERVAL+INITIAL_INTER][3])<<"; "<<int(poses[i*INTERVAL+INITIAL_INTER][11])<<endl;
        circle(traj, t_center, 1, Scalar(255, 0, 0), 2);

        // cout << t.t() << endl;
        // cout << "["<<poses[i][3] << ", " << poses[i][7] << ", " << poses[i][11] <<"]"<<endl;
        // cout<<-(t.at<float>(2))<<endl;
        // cout<<"pose:"<<poses[i][11]<<endl;
        imshow("Trajectory", traj);
        
        waitKey(1);
    }
        cout<<"map point_finish: "<<endl;
        for(int j = 0; j < tslam.all_mappoint.size();j++)
        {
            cout<<tslam.all_mappoint[j].pt3d<<", ";

        }
        cout<<endl;
    save_pcl(100,100,tslam.all_mappoint,"mp.pcd");

        waitKey(-1);
}