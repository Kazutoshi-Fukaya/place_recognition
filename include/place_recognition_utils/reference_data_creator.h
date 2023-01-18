#ifndef REFERENCE_DATA_CREATOR_H_
#define REFERENCE_DATA_CREATOR_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/utils.h>

#include "place_recognition_utils/inpaintor.h"

namespace place_recognition
{
class Data
{
public:
    Data() :
        equ_file_path(std::string("")),
        rgb_file_path(std::string("")),
        x(0.0), y(0.0), theta(0.0) {}

    Data(std::string _equ_file_path,std::string _rgb_file_path,
         double _x,double _y,double _theta) :
        equ_file_path(_equ_file_path), rgb_file_path(_rgb_file_path),
        x(_x), y(_y), theta(_theta) {}

    std::string equ_file_path;
    std::string rgb_file_path;
    double x;
    double y;
    double theta;
private:
};

class ReferenceData : public std::vector<Data>
{
public:
    void push_data(Data data) { this->emplace_back(data); }

private:
};

class ReferenceDataCreator
{
public:
    ReferenceDataCreator();
    ~ReferenceDataCreator();
    void process();

private:
    void equ_image_callback(const sensor_msgs::ImageConstPtr& msg);
    void rgb_image_callback(const sensor_msgs::ImageConstPtr& msg);
    void pose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg);

    void create_reference_data();

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // subscriber
    ros::Subscriber equ_sub_;
    ros::Subscriber rgb_sub_;
    ros::Subscriber pose_sub_;

    // inpaintor
    Inpaintor* inpaintor_;

    // buffer
	ReferenceData reference_data_;
    ros::Time start_time_;
    geometry_msgs::PoseStamped pose_;
    cv::Mat equ_img_;
    cv::Mat rgb_img_;
    int count_;

    // params
    std::string FILE_PATH_;
    int HZ_;

};

} // namespace place_recognition


#endif	// REFERENCE_DATA_CREATOR_H_