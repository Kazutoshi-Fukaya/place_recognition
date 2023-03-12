#include "place_recognition_utils/keypoints_recorder.h"

using namespace place_recognition;

KeypointsRecorder::KeypointsRecorder() :
    private_nh_("~"),
    detector_(cv::ORB::create()),
    count_(0)
{
    private_nh_.param("FILE_PATH",FILE_PATH_,{std::string("")});
    private_nh_.param("IS_RECORD",IS_RECORD_,{false});

    img_sub_ = nh_.subscribe("img_in",1,&KeypointsRecorder::image_callback,this);
    img_pub_ = nh_.advertise<sensor_msgs::Image>("img_out",1);
}

void KeypointsRecorder::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& ex){
        ROS_ERROR("Could not convert to color image");
        return;
    }

    // detect
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    detector_->detect(cv_ptr->image,keypoints);
    detector_->compute(cv_ptr->image,keypoints,descriptor);
    cv::Mat detected_img(cv_ptr->image.rows,cv_ptr->image.cols,cv_ptr->image.type());
    cv::drawKeypoints(cv_ptr->image,keypoints,detected_img,cv::Scalar(0.0,255.0,0),cv::DrawMatchesFlags::DEFAULT);

    // record
    if(IS_RECORD_ && count_%10 == 0){
        std::string rgb_img_name = "/rgb/image" + std::to_string(count_/10) + ".jpg";
        std::string orb_img_name = "/orb/image" + std::to_string(count_/10) + ".jpg";

        cv::imwrite(FILE_PATH_ + rgb_img_name,cv_ptr->image);
        cv::imwrite(FILE_PATH_ + orb_img_name,detected_img);

        count_++;
    }else count_++;


    // publish
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8",detected_img).toImageMsg();
    img_pub_.publish(img_msg);
}

void KeypointsRecorder::process() { ros::spin(); }

int main(int argc,char** argv)
{
    ros::init(argc,argv,"keypoints_recorder");
    KeypointsRecorder keypoints_recorder;
    keypoints_recorder.process();
    return 0;
}
