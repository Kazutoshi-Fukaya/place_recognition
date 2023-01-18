#ifndef KEYPOINTS_DETECTOR_H_
#define KEYPOINTS_DETECTOR_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "place_recognition_utils/inpaintor.h"

namespace place_recognition
{
class KeypointsDetector
{
public:
    KeypointsDetector();
    void process();

private:
    void image_callback(const sensor_msgs::ImageConstPtr& msg);

    void set_detector_mode(std::string mode);

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // subscriber
    ros::Subscriber img_sub_;

    // publisher
    ros::Publisher img_pub_;

    // inpaintor
    Inpaintor* inpaintor_;

    // detector
    cv::Ptr<cv::Feature2D> detector_;
};
}

#endif	// KEYPOINTS_DETECTOR_H_