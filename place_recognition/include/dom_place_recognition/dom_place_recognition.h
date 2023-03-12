#ifndef DOM_PLACE_RECOGNITION_H_
#define DOM_PLACE_RECOGNITION_H_

// ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/utils.h>

// utils
#include "dbow3/vocabulary/vocabulary.h"
#include "dbow3/database/database.h"
#include "place_recognition/images.h"
#include "dom_place_recognition/dom_place_recognition_interfaces.h"

namespace place_recognition
{
class DomPlaceRecognition
{
public:
    DomPlaceRecognition();
    ~DomPlaceRecognition();
    void process();

private:
    void set_detector_mode(std::string detector_mode);
    void load_reference_images(std::string reference_images_path,std::string image_mode);
    void calc_features(Image& image,std::string name,cv::Mat img);
    void create_database(std::string reference_images_path,std::string image_mode);
    void update_database();
    std::vector<std::string> split(std::string& input,char delimiter);

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // database
    dbow3::Database* database_;

    // interface
    DomPlaceRecognitionInterfaces* interfaces_;

    // detector
    cv::Ptr<cv::Feature2D> detector_;

    // Reference
    std::vector<Images> reference_images_;
    std::vector<Images> updated_reference_images_;

    // param
    std::string REFERENCE_IMAGES_PATH_;
    std::string IMAGE_MODE_;
    int HZ_;
};
} // namespace place_recognition

#endif  // DOM_PLACE_RECOGNITION_H_