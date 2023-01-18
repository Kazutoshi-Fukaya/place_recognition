#ifndef MULTI_PLACE_RECOGNITION_H_
#define MULTI_PLACE_RECOGNITION_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/utils.h>

#include "dbow3/vocabulary/vocabulary.h"
#include "dbow3/database/database.h"

#include "place_recognition/images.h"
#include "multi_place_recognition/place_recognition_interfaces.h"

namespace place_recognition
{
class MultiPlaceRecognition
{
public:
    MultiPlaceRecognition();
    ~MultiPlaceRecognition();
    void process();

private:
    void set_detector_mode(std::string detector_mode);
    void load_reference_images(std::string reference_images_path,std::string image_mode);
    void calc_features(Image& image,std::string name,cv::Mat img);
    void create_database(std::string reference_images_path,std::string image_mode);

    std::vector<std::string> split(std::string& input,char delimiter);

    void publish_result();

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // publisher
    ros::Publisher img_pub_;

    // database
    dbow3::Database* database_;

    // interface
    PlaceRecognitionInterfaces* interfaces_;

    // detector
    cv::Ptr<cv::Feature2D> detector_;

    // Reference
    std::vector<Images> reference_images_;

    // param
    std::string IMAGE_MODE_;
    bool IS_RECORD_;
    bool PUBLISH_IMAGE_;
    int HZ_;
};
} // namespace place_recognition

#endif  // MULTI_PLACE_RECOGNITION_H_
