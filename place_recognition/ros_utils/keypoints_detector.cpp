#include "place_recognition_utils/keypoints_detector.h"

using namespace place_recognition;

KeypointsDetector::KeypointsDetector() :
    private_nh_("~"),
    inpaintor_(new Inpaintor)
{
    // mode
    std::string mode;
    private_nh_.param("MODE",mode,{std::string("orb")});
    set_detector_mode(mode);

    // inpaintor params
    Pillars pillars;
    private_nh_.param("START_OF_FIRST_PILLAR",pillars.first.start,{150});
    private_nh_.param("END_OF_FIRST_PILLAR",pillars.first.end,{210});
    private_nh_.param("START_OF_SECOND_PILLAR",pillars.second.start,{420});
    private_nh_.param("END_OF_SECOND_PILLAR",pillars.second.end,{490});
    private_nh_.param("START_OF_THIRD_PILLAR",pillars.third.start,{800});
    private_nh_.param("END_OF_THIRD_PILLAR",pillars.third.end,{840});
    private_nh_.param("START_OF_FOURTH_PILLAR",pillars.fourth.start,{1060});
    private_nh_.param("END_OF_FOURTH_PILLAR",pillars.fourth.end,{1120});
    inpaintor_->set_params(pillars);

    img_sub_ = nh_.subscribe("img_in",1,&KeypointsDetector::image_callback,this);

    img_pub_ = nh_.advertise<sensor_msgs::Image>("img_out",1);
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

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    detector_->detect(inpainted_img,keypoints);
    detector_->compute(inpainted_img,keypoints,descriptor);
    cv::Mat detected_img(cv_ptr->image.rows,cv_ptr->image.cols,cv_ptr->image.type());
    cv::drawKeypoints(inpainted_img,keypoints,detected_img,cv::Scalar(0.0,255.0,0),cv::DrawMatchesFlags::DEFAULT);

    // compare
    cv::Mat space(cv::Mat::zeros(20,resized_img.cols,CV_8UC3));
    cv::Mat base(3*resized_img.rows + 2*space.rows,resized_img.cols,CV_8UC3);
    cv::Mat roi_1(base,cv::Rect(0,0,resized_img.cols,resized_img.rows));
    resized_img.copyTo(roi_1);
    cv::Mat roi_2(base,cv::Rect(0,resized_img.rows,space.cols,space.rows));
    space.copyTo(roi_2);
    cv::Mat roi_3(base,cv::Rect(0,resized_img.rows + space.rows,inpainted_img.cols,inpainted_img.rows));
    inpainted_img.copyTo(roi_3);
    cv::Mat roi_4(base,cv::Rect(0,2*resized_img.rows + space.rows,space.cols,space.rows));
    space.copyTo(roi_4);
    cv::Mat roi_5(base,cv::Rect(0,2*resized_img.rows + 2*space.rows,detected_img.cols,detected_img.rows));
    detected_img.copyTo(roi_5);

    // publish
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8",base).toImageMsg();
    img_pub_.publish(img_msg);
}

void KeypointsDetector::set_detector_mode(std::string mode)
{
    if(mode == "orb") detector_ = cv::ORB::create();
    else if(mode == "brisk") detector_ = cv::BRISK::create();
    else if(mode == "akaze") detector_ = cv::AKAZE::create();
    else{
        ROS_WARN("No applicable mode. Please select 'orb', 'brisk' or 'akaze'");
        ROS_INFO("Set 'orb");
        detector_ = cv::ORB::create();
    }
}

void KeypointsDetector::process() { ros::spin(); }

int main(int argc,char** argv)
{
    ros::init(argc,argv,"keypoints_detector");
    KeypointsDetector keypoints_detector;
    keypoints_detector.process();
    return 0;
}
