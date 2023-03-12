#include "dom_place_recognition/dom_place_recognition_interface.h"

using namespace dbow3;
using namespace place_recognition;

DomPlaceRecognitionInterface::DomPlaceRecognitionInterface() 
{
	ROS_INFO("'dom_place_recognition_interface' has not received 'nh', 'name' and 'database'.");
}

DomPlaceRecognitionInterface::DomPlaceRecognitionInterface(ros::NodeHandle nh,std::string name,dbow3::Database* database,std::string image_mode,bool is_record,bool is_vis) :
	database_(database), updated_database_(database), 
	engine_(seed_()), start_time_(ros::Time::now()), img_count_(0), is_updated_(false),
	IMAGE_MODE_(image_mode), MAP_FRAME_ID_(std::string("map")),
	IS_RECORD_(is_record), IS_VIS_(is_vis)
{
	// image subscriber
    std::string img_sub_topic_name;
    if(IMAGE_MODE_ == "rgb") img_sub_topic_name = name + "/camera/color/image_rect_color";
    else if(IMAGE_MODE_ == "equ") img_sub_topic_name = name + "/equirectangular/image_raw";
    img_sub_ = nh.subscribe(img_sub_topic_name,1,&DomPlaceRecognitionInterface::image_callback,this);

	// object detection subscriber
	std::string od_topic_name = name + "/object_positions";
	od_sub_ = nh.subscribe(od_topic_name,1,&DomPlaceRecognitionInterface::od_callback,this);
	
	// pose subscriber
	std::string pose_topic_name = name + "/pose";
	pose_sub_ = nh.subscribe(pose_topic_name,1,&DomPlaceRecognitionInterface::pose_callback,this);

	// place recognition pose publisher (result)
    std::string pr_pose_topic_name = name + "/pr_pose";
    pr_pose_pub_ = nh.advertise<place_recognition_msgs::PoseStamped>(pr_pose_topic_name,1);

	// for visualization
	if(IS_VIS_){
		std::string vis_pose_topic_name = name + "/vis_pr_pose";
		vis_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>(vis_pose_topic_name,1);
	}
}

void DomPlaceRecognitionInterface::set_reference_image_path(std::string path) { REFERENCE_IMAGES_PATH_ = path; }

void DomPlaceRecognitionInterface::set_detector(cv::Ptr<cv::Feature2D> detector) { detector_ = detector; }

void DomPlaceRecognitionInterface::set_reference_images(std::vector<Images> reference_images) { reference_images_ = reference_images; }

void DomPlaceRecognitionInterface::set_database(dbow3::Database* database) 
{
	delete database_; 
	database_ = database; 
}

void DomPlaceRecognitionInterface::set_dom_params(std::vector<DomParam> dom_params) { dom_params_ = dom_params; }

void DomPlaceRecognitionInterface::update_param(std::vector<Images> updated_reference_images,dbow3::Database* updated_database)
{
	updated_reference_images_ = updated_reference_images;
	delete updated_database_;
	updated_database_ = updated_database;
	img_count_ = 0;
}

std::vector<Images> DomPlaceRecognitionInterface::get_stored_images() { return stored_images_; }

cv::Mat DomPlaceRecognitionInterface::get_input_img() { return input_img_; }

cv::Mat DomPlaceRecognitionInterface::get_output_img() { return output_img_; }

void DomPlaceRecognitionInterface::image_callback(const sensor_msgs::ImageConstPtr& msg)
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

	if(!od_.object_position.empty()){
		ObjectNameList name_list;
		for(const auto &od_pos : od_.object_position){
			name_list.add_name(od_pos.Class);
		}

		double time = (ros::Time::now() - start_time_).toSec();
		std::uniform_real_distribution<> dist(0.0,1.0);
		for(const auto &nl : name_list){
			DomParam dom_param;
			for(const auto &dp : dom_params_){
				if(nl == dp.name) dom_param = dp;
			}
			double p_0 = std::exp(-1.0*(double)(dom_param.appearance_count+dom_param.disappearance_count)/(double)dom_param.object_size/time);
			if(dist(engine_) > p_0){
				Image rgb;
				std::vector<cv::KeyPoint> keypoints;
    			cv::Mat descriptors;
    			detector_->detectAndCompute(input_img_,cv::Mat(),keypoints,descriptors);
				rgb.set_params(std::to_string(img_count_),input_img_,keypoints,descriptors);
				Images images(pose_.pose.position.x,pose_.pose.position.y,tf2::getYaw(pose_.pose.orientation));
				images.set_rgb_image(rgb);
				stored_images_.emplace_back(images);	
				img_count_++;
				return;
			}
		}
	}

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

	ros::Time now_time = ros::Time::now();
	place_recognition_msgs::PoseStamped pr_pose;
    pr_pose.header.frame_id = MAP_FRAME_ID_;
    pr_pose.header.stamp = now_time;
    pr_pose.score = ret.at(0).score;
    pr_pose.x = reference_images_.at(id).x;
    pr_pose.y = reference_images_.at(id).y;
    pr_pose.theta = reference_images_.at(id).theta;
    pr_pose_pub_.publish(pr_pose);

	if(IS_VIS_){
		geometry_msgs::PoseStamped vis_pose;
		vis_pose.header.frame_id = MAP_FRAME_ID_;
		vis_pose.header.stamp = now_time;
		vis_pose.pose.position.x = reference_images_.at(id).x;
		vis_pose.pose.position.y = reference_images_.at(id).y;
		vis_pose.pose.position.z = 0.0;
		tf2::Quaternion tf_q;
        tf_q.setRPY(0.0,0.0,reference_images_.at(id).theta);
		vis_pose.pose.orientation.x = tf_q.x();
		vis_pose.pose.orientation.y = tf_q.y();
		vis_pose.pose.orientation.z = tf_q.z();
		vis_pose.pose.orientation.w = tf_q.w();
		vis_pose_pub_.publish(vis_pose);
	}
}

void DomPlaceRecognitionInterface::pose_callback(const geometry_msgs::PoseStampedConstPtr& msg) { pose_ = *msg; }

void DomPlaceRecognitionInterface::od_callback(const object_detector_msgs::ObjectPositionsConstPtr& msg) { od_ = *msg; }