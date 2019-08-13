#include "Tslam.hpp"
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

int main()
{
    Tslam tslam;
    tslam.initialize();

    // for(int i = 0;i<10;i++)
    // {
    //     tslam.addframe();
    //     // cout<<"map point: "<<endl;
    //     // for(int j = 0; j < tslam.all_mappoint.size();j++)
    //     // {
    //     //     cout<<tslam.all_mappoint[j].pt3d<<", ";

    //     // }
    //     // cout<<endl;
    //     cout<<"mp_size: "<<tslam.all_mappoint.size()<<endl;
    // }
    
    Mat traj = Mat::zeros(600, 600, CV_8UC3);
    vector<vector<float>> poses = get_Pose("/data/SLAM/dataset/poses/00.txt");

    for (int i = 0; i < 7; i++)
    {
        tslam.addframe();
        Mat x = tslam.last_frame.keyframe_.t;
        Point2f center = Point2f((int(x.at<double>(0, 0))) + 300, (int(x.at<double>(2, 0))) + 100);
        cout<<"pred: "<<int(x.at<double>(0, 0))<<"; "<<int(x.at<double>(2, 0))<<endl;
        circle(traj, center, 1, Scalar(0, 0, 255), 2);

        Point2f t_center = Point2f(int(poses[i+2][3]) + 300, int(poses[i+2][11]) + 100);
        cout<<"gt: "<<int(poses[i+2][3])<<"; "<<int(poses[i+2][11])<<endl;
        circle(traj, t_center, 1, Scalar(255, 0, 0), 2);

        // cout << t.t() << endl;
        // cout << "["<<poses[i][3] << ", " << poses[i][7] << ", " << poses[i][11] <<"]"<<endl;
        // cout<<-(t.at<float>(2))<<endl;
        // cout<<"pose:"<<poses[i][11]<<endl;
        imshow("Trajectory", traj);
        
        waitKey(1);
    }
        waitKey(-1);
}