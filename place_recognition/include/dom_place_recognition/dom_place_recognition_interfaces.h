#ifndef DOM_PLACE_RECOGNITION_INTERFACES_H_
#define DOM_PLACE_RECOGNITION_INTERFACES_H_

// Custom msg
#include "dom_estimator_msgs/Doms.h"

// utils
#include "dom_place_recognition/dom_place_recognition_interface.h"

namespace place_recognition
{
class DomPlaceRecognitionInterfaces : public std::vector<DomPlaceRecognitionInterface*>
{
public:
	DomPlaceRecognitionInterfaces();
	DomPlaceRecognitionInterfaces(ros::NodeHandle nh,ros::NodeHandle private_nh,dbow3::Database* database);
	DomPlaceRecognitionInterfaces(ros::NodeHandle nh,ros::NodeHandle private_nh,dbow3::Database* database,std::string image_mode,bool is_record,bool is_vis);

	void set_reference_image_path(std::string path);
	void set_detector(cv::Ptr<cv::Feature2D> detector);
	void set_reference_images(std::vector<Images> reference_images);
	void update_params(std::vector<Images> updated_reference_images,dbow3::Database* updated_database);
	bool stocked_enough_images(std::vector<Images>& collected_images);

private:
	void doms_callback(const dom_estimator_msgs::DomsConstPtr& msg);
	void init(ros::NodeHandle nh,ros::NodeHandle private_nh,dbow3::Database* database,std::string image_mode,bool is_record, bool is_vis);

	// subscriber
	ros::Subscriber doms_sub_;
};
} // namespace place_recognition

#endif	// DOM_PLACE_RECOGNITION_INTERFACES_H_