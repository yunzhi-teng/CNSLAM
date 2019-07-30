#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/opencv.hpp"
#include <fstream>
using namespace cv;
using namespace std;
char* raw_path = "/data/SLAM/dataset/kitti/00/image_0/%06d.png";
Mat getimage(int i)
{
    char path[50];
	sprintf(path, raw_path, i);
    Mat img =imread(path);
    return img;
}
Mat VO(int image_index,Mat& x);
std::vector<vector<float>> get_Pose(const std::string& path) {

  std::vector<vector<float>> poses;
  ifstream myfile(path);
  string line;
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
          char * dup = strdup(line.c_str());
   		  char * token = strtok(dup, " ");
   		  std::vector<float> v;
	   	  while(token != NULL){
	        	v.push_back(atof(token));
	        	token = strtok(NULL, " ");
	    	}
	    	poses.push_back(v);
	    	free(dup);
    }
    myfile.close();
  } else {
  	cout << "Unable to open file"; 
  }	

  return poses;

}
int main()
{
    int image_index = 0;
    Mat traj = Mat::zeros(600, 600, CV_8UC3);
    vector<vector<float>> poses = get_Pose("/data/SLAM/dataset/poses/00.txt");
    Mat x = Mat::zeros(3,1,CV_64F);
    // Mat R = Mat::zeros(3,3,CV_64F);
    for(int i = 0;i<100;i++)
    {
        Mat tt = VO(image_index,x);
        image_index++;

        Point2f center = Point2f((int(x.at<double>(0,0))) + 300, -(int(x.at<double>(2,0))) + 100);
        circle(traj, center ,1, Scalar(0,0,255), 2);

        Point2f t_center = Point2f(int(poses[i][3]) + 300, int(poses[i][11]) + 100);
        circle(traj, t_center,1, Scalar(255,0,0), 2);

		// cout << t.t() << endl;
		// cout << "["<<poses[i][3] << ", " << poses[i][7] << ", " << poses[i][11] <<"]"<<endl;
        // cout<<-(t.at<float>(2))<<endl;
        cout<<"pose:"<<poses[i][11]<<endl;
        imshow( "Trajectory", traj);
        waitKey(1);

    }
}
Mat VO(int image_index,Mat & x)
{
    Mat desc1,desc2;
    Mat img1 = getimage(image_index);
    Mat img2 = getimage(image_index+1);
    vector<KeyPoint> keypoints1,keypoints2;
    vector<DMatch> matches;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    detector->detect(img1,keypoints1);
    detector->detect(img2,keypoints2);

    descriptor->compute(img1, keypoints1,desc1);
    descriptor->compute(img2, keypoints2, desc2);

    vector<DMatch> match;

    matcher->match(desc1,desc2,match);
    double min_dist = 8888, max_dist = 0;
    for (int i = 0; i<desc1.rows;i++)
    {
        double dist = match[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist < max_dist) max_dist = dist;

    }
    for ( int i = 0; i < desc1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
    cout<<"total matches:"<<matches.size()<<endl;
    Mat R,t;
    Mat K = (Mat_<double> (3,3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,0.000000000000e+00, 7.188560000000e+02 ,1.852157000000e+02,0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);

    vector<Point2f> points1,points2;
    for( int i = 0;i < (int)matches.size();i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);

    }

    Point2d principal_point(6.071928000000e+02,1.852157000000e+02);
    double focal_length = 7.188560000000e+02;
    Mat essential_mat;
    Mat confi;
    essential_mat = findEssentialMat(points1,points2,focal_length, principal_point,8,0.99999,3,confi);

    recoverPose(essential_mat,points1,points2,R,t,focal_length,principal_point);
    cout<< "t:"<<t.t()<<endl;
    cout<< "R: "<<R<<endl;
    double s = sum(confi)[0];
    if(s / confi.rows/confi.cols >0.95) 
        x = R*x + t;
    cout<<"to:"<<x.t()<<endl;
    cout<<"to2: "<<(x.at<double>(2,0)) <<endl;
    return t;

}