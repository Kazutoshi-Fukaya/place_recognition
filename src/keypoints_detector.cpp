#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "utils/inpaintor.h"

namespace place_recognition
{
class KeypointsDetector
{
public:
	KeypointsDetector();
	void process();

private:
	void image_callback(const sensor_msgs::ImageConstPtr& msg);

	// node handler
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;

	// subscriber
	ros::Subscriber img_sub_;

	// inpaintor
	Inpaintor* inpaintor_;
};
}

using namespace place_recognition;

KeypointsDetector::KeypointsDetector() :
	private_nh_("~"),
	inpaintor_(new Inpaintor)
{
	img_sub_ = nh_.subscribe("img_in",1,&KeypointsDetector::image_callback,this);
}

void KeypointsDetector::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& ex){
        ROS_ERROR("Could not convert to color image");
        return;
    }

	cv::Mat resized_img(cv_ptr->image.rows,cv_ptr->image.cols,cv_ptr->image.type());
	resized_img = cv_ptr->image;

	cv::Mat inpainted_img(cv_ptr->image.rows,cv_ptr->image.cols,cv_ptr->image.type());
	inpainted_img = cv_ptr->image;

	inpaintor_->resize_img(resized_img);
	inpaintor_->inpaint_img(cv_ptr->image,inpainted_img);

	std::cout << "Resize  Img: (" << resized_img.cols << "," << resized_img.rows << ")" << std::endl;
	std::cout << "Inpaint Img: (" << inpainted_img.cols << "," << inpainted_img.rows << ")" << std::endl;
	std::cout << std::endl;

	// TO DO (convine)

}

void KeypointsDetector::process() { ros::spin(); }

int main(int argc,char** argv)
{
	ros::init(argc,argv,"keypoints_detector");
	KeypointsDetector keypoints_detector;
	keypoints_detector.process();
	return 0;
}