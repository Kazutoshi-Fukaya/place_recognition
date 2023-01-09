#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "utils/inpaintor.h"

namespace place_recognition
{
class ImageCreator
{
public:
	ImageCreator();
	void process();

private:
	void image_callback(const sensor_msgs::ImageConstPtr& msg);
	double get_time();

	// node handler
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;

	// subscriber
	ros::Subscriber img_sub_;

	// inpaintor
	Inpaintor* inpaintor_;

	// buffer
	ros::Time start_time_;
	int count_;

	// params
	std::string FILE_PATH_;
};
}

using namespace place_recognition;

ImageCreator::ImageCreator() :
	private_nh_("~"),
	inpaintor_(new Inpaintor),
	start_time_(ros::Time::now()), count_(0)
{
	private_nh_.param("FILE_PATH",FILE_PATH_,{std::string("")});

	img_sub_ = nh_.subscribe("img_in",1,&ImageCreator::image_callback,this);
}

void ImageCreator::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& ex){
        ROS_ERROR("Could not convert to color image");
        return;
    }

	cv::Mat inpainted_img(cv_ptr->image.rows,cv_ptr->image.cols,cv_ptr->image.type());
	inpainted_img = cv_ptr->image;
	inpaintor_->inpaint_img(cv_ptr->image,inpainted_img);

	if(count_%10 == 0){
		std::string file_name = FILE_PATH_ + std::to_string(count_) + "img.jpg";
		cv::imwrite(file_name,inpainted_img);
	}
	count_++;
}

double ImageCreator::get_time()
{
	return (ros::Time::now() - start_time_).toSec();
}

void ImageCreator::process() { ros::spin(); }

int main(int argc,char** argv)
{
	ros::init(argc,argv,"image_creator");
	ImageCreator image_creator;
	image_creator.process();
	return 0;
}