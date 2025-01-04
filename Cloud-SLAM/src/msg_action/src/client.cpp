#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <actionlib/client/simple_action_client.h>
// custom
#include "msg_action/Sequence.h"
#include "msg_action/CloudSlamAction.h"

typedef actionlib::SimpleActionClient<msg_action::CloudSlamAction> Client;


void ReadImagesAndTimestamps(cv::String folder_path, int len, std::vector<sensor_msgs::Image> &img, std::vector<double> &tstamps);
sensor_msgs::CameraInfo getCameraInfo(void);

// cd /media/edward/Data/Workspace/LAB/edgecloud/test_ws/devel/lib/msg_action
// /data/SLAMdatasets/tum/rgbd_dataset_freiburg2_pioneer_360/rgb
int main(int argc, char **argv)
{

    cv::String folder_path = argv[2];
    std::vector<cv::Mat> img;
    std::vector<double> tstamps;
    char* len = argv[1];

    ros::init(argc, argv, "edge");
    ros::NodeHandle n;

    Client ac("cloudslam", true);
    ac.waitForServer();

    msg_action::CloudSlamGoal goal;
    std::cout << "start reading images" << std::endl;
    ReadImagesAndTimestamps(folder_path, atoi(len), goal.sequence.images, goal.sequence.timestamps);
    std::cout << len << " images" << std::endl;
    goal.sequence.camera = getCameraInfo();
    
    std::cout << "=========" << std::endl;
    ac.sendGoalAndWait(goal);
    std::cout << ac.getResult()->map.key_frames.size() << std::endl;
    std::cout << ac.getResult()->map.map_points.size() << std::endl;

    return 0;
}

void ReadImagesAndTimestamps(cv::String folder_path, int len, std::vector<sensor_msgs::Image> &img, std::vector<double> &tstamps)
{
    std::vector<cv::String> fn;
    cv::glob(folder_path, fn, false);
    int count = fn.size(); //number of png files in images folder

    int startpos = folder_path.length()+1;
    int n = fn[0].length() - startpos - 4; //length of a timestamp
    for (int i = 0; i < len; i++)
    {
        cv::Mat img_i = cv::imread(fn[i], cv::IMREAD_COLOR);
        sensor_msgs::ImagePtr p = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_i).toImageMsg();
        img.emplace_back(*p);

        std::string tp = fn[i].substr(startpos, n);
        // std::cout << tp << std::endl;
        tstamps.emplace_back(std::stod(tp));
    }
}

sensor_msgs::CameraInfo getCameraInfo(void)
{
    sensor_msgs::CameraInfo cam;

    std::vector<double> D{0.000094, -0.011701, 0.000383, -0.000507, 0.000000};
    boost::array<double, 9> K = {
        535.4, 0.000000, 320.1,
        0.000000, 539.2, 247.6,
        0.000000, 0.000000, 1.000000  
    };
    
    boost::array<double, 12> P = {
        402.124725, 0.000000, 335.482488, 0.000000,
        0.000000, 403.765045, 250.954855, 0.000000,
        0.000000, 0.000000, 1.000000, 0.000000
    };
    boost::array<double, 9> r = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    cam.height = 480;
    cam.width = 640;
    cam.distortion_model = "plumb_bob";
    cam.D = D;
    cam.K = K;
    cam.P = P;
    cam.R = r;
    cam.binning_x = 0;
    cam.binning_y = 0;
    cam.header.frame_id = "camera";
    cam.header.stamp = ros::Time::now();
    cam.header.stamp.nsec = 0;
    return cam;
}