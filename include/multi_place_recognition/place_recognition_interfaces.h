#ifndef PLACE_RECOGNITION_INTERFACES_H_
#define PLACE_RECOGNITION_INTERFACES_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2/utils.h>
#include <opencv2/opencv.hpp>

#include "dbow3/vocabulary/vocabulary.h"
#include "dbow3/database/database.h"
#include "multi_place_recognition/match_recorder.h"

#include "place_recognition_msgs/PoseStamped.h"

namespace place_recognition
{
class PlaceRecognitionInterface
{
public:
	PlaceRecognitionInterface() {}
	PlaceRecognitionInterface(ros::NodeHandle nh,std::string name,dbow3::Database* database,std::string image_mode,bool is_record) :
		nh_(nh), database_(database), start_time_(ros::Time::now()), IMAGE_MODE_(image_mode), IS_RECORD_(is_record)
	{
		// image subscriber
		std::string img_sub_topic_name;
		if(IMAGE_MODE_ == "rgb") img_sub_topic_name = name + "/camera/color/image_rect_color";
		else if(IMAGE_MODE_ == "equ") img_sub_topic_name = name + "/equirectangular/image_raw";
		img_sub_ = nh_.subscribe(img_sub_topic_name,1,&PlaceRecognitionInterface::image_callback,this);
		
		// pose subscriber (for ground truth)
		if(IS_RECORD_){
			match_recorder_ = new MatchRecorder();
			std::string pose_sub_topic_name = name + "/amcl_pose";
			pose_sub_ = nh_.subscribe(pose_sub_topic_name,1,&PlaceRecognitionInterface::pose_callback,this);
		}

		// pose publisher (result)
		std::string pose_topic_name = name + "/pr_pose";
		pose_pub_ = nh_.advertise<place_recognition_msgs::PoseStamped>(pose_topic_name,1);

	}

	void set_reference_image_path(std::string path) { REFERENCE_IMAGES_PATH_ = path; }
	void set_detector(cv::Ptr<cv::Feature2D> detector) { detector_ = detector; }	
	void set_reference_images(std::vector<Images> reference_images) { reference_images_ = reference_images; }


	cv::Mat get_input_img() { return input_img_; }
	cv::Mat get_output_img() { return output_img_; }

	// match recorder
	MatchRecorder* match_recorder_;

private:
	void pose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
	{
		ref_pose_.header = msg->header;
		ref_pose_.pose = msg->pose.pose;
	}

	void image_callback(const sensor_msgs::ImageConstPtr& msg)
	{
		cv_bridge::CvImagePtr cv_ptr;
		try{
			cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
		}
    	catch(cv_bridge::Exception& ex){
        	ROS_ERROR("Could not convert to color image");
        	return;
    	}

		input_img_ = cv_ptr->image;
		if(input_img_.empty()) return;

		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		detector_->detectAndCompute(cv_ptr->image,cv::Mat(),keypoints,descriptors);

		dbow3::QueryResults ret;
		database_->query(descriptors,ret,4);
    	if(ret.empty()) return;

		int id = ret.at(0).id;
		std::string name;
		if(IMAGE_MODE_ == "rgb") name = REFERENCE_IMAGES_PATH_ + "rgb/image" + std::to_string(id) + ".jpg";
		else if(IMAGE_MODE_ ==  "equ") name = REFERENCE_IMAGES_PATH_ + "equ/image" + std::to_string(id) + ".jpg";
		cv::Mat output_img = cv::imread(name);
		if(output_img.empty()) return;
		output_img_ = output_img;

		if(IS_RECORD_){
			MatchPosition est_pos, ref_pos;
			double now_time = (ros::Time::now() - start_time_).toSec();
			est_pos.x = reference_images_.at(id).x;
			est_pos.y = reference_images_.at(id).y;
			est_pos.theta = reference_images_.at(id).theta;
			ref_pos.x = ref_pose_.pose.position.x;
			ref_pos.y = ref_pose_.pose.position.y;
			ref_pos.theta = tf2::getYaw(ref_pose_.pose.orientation);
			match_recorder_->add_match_record(now_time,est_pos,ref_pos);
		}

		// sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8",output_img).toImageMsg();
		// img_pub_.publish(img_msg);

		place_recognition_msgs::PoseStamped pose;
		pose.header.frame_id = "map";
		pose.header.stamp = ros::Time::now();
		pose.score = ret.at(0).score;
		pose.x = reference_images_.at(id).x;
		pose.y = reference_images_.at(id).y;
		pose.theta = reference_images_.at(id).theta;

		pose_pub_.publish(pose);
	}

	// node handler
    ros::NodeHandle nh_;

    // subscriber
    ros::Subscriber img_sub_;
	ros::Subscriber pose_sub_;
	
	// Publisher
	ros::Publisher pose_pub_;

    // database
	dbow3::Database* database_;

	// detector
    cv::Ptr<cv::Feature2D> detector_;

	// reference_images
	std::vector<Images> reference_images_;

	// buffer
	geometry_msgs::PoseStamped ref_pose_;
	ros::Time start_time_;
	cv::Mat input_img_;
	cv::Mat output_img_;

	// param
	std::string IMAGE_MODE_;
	std::string REFERENCE_IMAGES_PATH_;
	bool IS_RECORD_;
};

class PlaceRecognitionInterfaces : public std::vector<PlaceRecognitionInterface*>
{
public:
	PlaceRecognitionInterfaces() {}
	PlaceRecognitionInterfaces(ros::NodeHandle _nh,ros::NodeHandle _private_nh,dbow3::Database* _database,std::string _image_mode,bool _is_record) :
		nh_(_nh), private_nh_(_private_nh), database_(_database), image_mode_(_image_mode), is_record_(_is_record) { init(); }

	void set_reference_image_path(std::string path)
	{
		for(size_t i = 0; i < this->size(); i++){
			this->at(i)->set_reference_image_path(path);
		}		
	}

	void set_detector(cv::Ptr<cv::Feature2D> detector)
	{
		for(size_t i = 0; i < this->size(); i++){
			this->at(i)->set_detector(detector);
		}
	}

	void set_reference_images(std::vector<Images> reference_images)
	{
		reference_images_ = reference_images;
		for(size_t i = 0; i < this->size(); i++){
			this->at(i)->set_reference_images(reference_images);
		}
	}

	cv::Mat get_img()
	{
		if(image_mode_ == "rgb"){
			if(this->at(0)->get_input_img().empty()) return cv::Mat(100,100,CV_8UC3);
			
			int rows = this->at(0)->get_input_img().rows;
			int cols = this->at(0)->get_input_img().cols;
			cv::Mat space_1(cv::Mat::zeros(rows,50,CV_8UC3));
			cv::Mat space_2(cv::Mat::zeros(50,cols*this->size() + space_1.cols*((int)this->size() - 1),CV_8UC3));
			cv::Mat base(2.0*rows + space_2.rows,cols*this->size() + space_1.cols*((int)this->size() - 1),CV_8UC3);
			for(size_t i = 0; i < this->size(); i++){
				cv::Mat input_img = this->at(i)->get_input_img();
				if(input_img.empty()) input_img = cv::Mat::zeros(rows,cols,CV_8UC3);
				cv::Mat roi_1(base,cv::Rect(i*(input_img.cols + space_1.cols),0,input_img.cols,input_img.rows));
				input_img.copyTo(roi_1);

				cv::Mat output_img = this->at(i)->get_output_img();
				if(output_img.empty()) output_img = cv::Mat::zeros(rows,cols,CV_8UC3);
				cv::Mat roi_2(base,cv::Rect(i*(input_img.cols + space_1.cols),rows + space_2.rows,output_img.cols,output_img.rows));
				output_img.copyTo(roi_2);

				if((int)i != (int)this->size() - 1){
					cv::Mat roi_3(base,cv::Rect((i + 1)*input_img.cols + i*space_1.cols,0,space_1.cols,space_1.rows));
					space_1.copyTo(roi_3);
				}
			}
			cv::Mat roi_4(base,cv::Rect(0,rows,space_2.cols,space_2.rows));
			space_2.copyTo(roi_4);

			return base;
		}
	}

private:
	void init()
	{
		this->clear();
		std::string robot_list_name;
		private_nh_.param("ROBOT_LIST",robot_list_name,{std::string("robot_list")});
		XmlRpc::XmlRpcValue robot_list;
		if(!private_nh_.getParam(robot_list_name.c_str(),robot_list)){
			ROS_ERROR("Cloud not load %s", robot_list_name.c_str());
			return;
		}
		
		ROS_ASSERT(robot_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
		this->resize(robot_list.size());
		for(int i = 0; i < (int)robot_list.size(); i++){
			if(!robot_list[i]["name"].valid()){
				ROS_ERROR("%s is valid", robot_list_name.c_str());
				return;
			}
			if(robot_list[i]["name"].getType() == XmlRpc::XmlRpcValue::TypeString){
				std::string name = static_cast<std::string>(robot_list[i]["name"]);
				this->at(i) = new PlaceRecognitionInterface(nh_,name,database_,image_mode_,is_record_);
        	}
    	}
	}

	// node handler
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;

	// database
	dbow3::Database* database_;

	// reference images
	std::vector<Images> reference_images_;

	// mode
	std::string image_mode_;
	bool is_record_;
};
} // namespace place_recognition

#endif	// PLACE_RECOGNITION_INTERFACES_H_