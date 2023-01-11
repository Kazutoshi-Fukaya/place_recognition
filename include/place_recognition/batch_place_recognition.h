#ifndef BATCH_PLACE_RECOGNITION_H_
#define BATCH_PLACE_RECOGNITION_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/utils.h>

#include "dbow3/vocabulary/vocabulary.h"
#include "dbow3/database/database.h"
#include "place_recognition/images.h"

namespace place_recognition
{
class BatchPlaceRecognition
{
public:
    BatchPlaceRecognition();
    void process();

private:
    void load_reference_images();
    void calc_features(Image& image,std::string name,cv::Mat img);

    void create_database();

    std::vector<std::string> split(std::string& input,char delimiter);

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // database
    dbow3::Database db_;

    // detector
    cv::Ptr<cv::Feature2D> detector_;

    // Reference
    std::vector<Images> reference_images_;

    // buffer

    // param
    std::string DIR_PATH_;
    std::string REFERENCE_FILE_PATH_;
	std::string QUERY_FILE_PATH_;
	std::string MODE_;
};
} // namespace place_recognition

#endif	// BATCH_PLACE_RECOGNITION_H_