#ifndef DOM_PLACE_RECOGNITION_INTERFACE_H_
#define DOM_PLACE_RECOGNITION_INTERFACE_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2/utils.h>

// Custom msg
#include "place_recognition_msgs/PoseStamped.h"
#include "object_detector_msgs/ObjectPositions.h"

// opencv
#include <opencv2/opencv.hpp>

// c++
#include <random>

// utils
#include "dbow3/vocabulary/vocabulary.h"
#include "dbow3/database/database.h"
#include "place_recognition/images.h"
#include "dom_place_recognition/dom_param.h"
#include "dom_place_recognition/object_name_list.h"

namespace place_recognition
{
class DomPlaceRecognitionInterface
{
public:
	DomPlaceRecognitionInterface();
	DomPlaceRecognitionInterface(ros::NodeHandle nh,std::string name,dbow3::Database* database,std::string image_mode,bool is_record,bool is_vis);

	void set_reference_image_path(std::string path);
    void set_detector(cv::Ptr<cv::Feature2D> detector);
    void set_reference_images(std::vector<Images> reference_images);
	void set_database(dbow3::Database* database);
    void set_dom_params(std::vector<DomParam> dom_params);
    void update_param(std::vector<Images> updated_reference_images,dbow3::Database* updated_database);

    std::vector<Images> get_stored_images();
	cv::Mat get_input_img();
    cv::Mat get_output_img();

private:
	void image_callback(const sensor_msgs::ImageConstPtr& msg);
    void pose_callback(const geometry_msgs::PoseStampedConstPtr& msg);
    void od_callback(const object_detector_msgs::ObjectPositionsConstPtr& msg);

    // subscriber
    ros::Subscriber img_sub_;
    ros::Subscriber pose_sub_;
    ros::Subscriber od_sub_;

    // Publisher
    ros::Publisher pr_pose_pub_;
	ros::Publisher vis_pose_pub_;

    // database
    dbow3::Database* database_;
    dbow3::Database* updated_database_;

    // random
    std::random_device seed_;
    std::mt19937 engine_;

    // dom_params
    std::vector<DomParam> dom_params_;

    // detector
    cv::Ptr<cv::Feature2D> detector_;

    // images
    std::vector<Images> reference_images_;
    std::vector<Images> updated_reference_images_;
    std::vector<Images> stored_images_;

    // buffer
    ros::Time start_time_;
    geometry_msgs::PoseStamped pose_;
    object_detector_msgs::ObjectPositions od_;
    cv::Mat input_img_;
    cv::Mat output_img_;
    bool is_updated_;
    int img_count_;

    // param
    std::string IMAGE_MODE_;
    std::string REFERENCE_IMAGES_PATH_;
	std::string MAP_FRAME_ID_;
    bool IS_RECORD_;
	bool IS_VIS_;	// for visualization
};
} // namespace place_recognition

#endif	// DOM_PLACE_RECOGNITION_INTERFACE_H_