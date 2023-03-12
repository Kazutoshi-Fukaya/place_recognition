#ifndef KEYPOINTS_RECORDER_H_
#define KEYPOINTS_RECORDER_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

// opencv
#include <opencv2/opencv.hpp>

namespace place_recognition
{
class KeypointsRecorder
{
public:
    KeypointsRecorder();
    void process();

private:
    void image_callback(const sensor_msgs::ImageConstPtr& msg);

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // subscriber
    ros::Subscriber img_sub_;

    // publisher
    ros::Publisher img_pub_;

    // detector
    cv::Ptr<cv::Feature2D> detector_;

    // buffer
    int count_;

    // param
    std::string FILE_PATH_;
    bool IS_RECORD_;
};
} // namespace place_recognition

#endif  // KEYPOINTS_RECORDER_H_
