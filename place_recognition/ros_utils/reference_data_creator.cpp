#include "place_recognition_utils/reference_data_creator.h"

using namespace place_recognition;

ReferenceDataCreator::ReferenceDataCreator() :
    private_nh_("~"),
    inpaintor_(new Inpaintor()),
    start_time_(ros::Time::now()),
    count_(0)
{
    private_nh_.param("FILE_PATH",FILE_PATH_,{std::string("")});
    private_nh_.param("HZ",HZ_,{10});

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

    equ_sub_ = nh_.subscribe("equ_in",1,&ReferenceDataCreator::equ_image_callback,this);
    rgb_sub_ = nh_.subscribe("rgb_in",1,&ReferenceDataCreator::rgb_image_callback,this);
    pose_sub_ = nh_.subscribe("pose_in",1,&ReferenceDataCreator::pose_callback,this);
}

ReferenceDataCreator::~ReferenceDataCreator()
{
    std::string save_file_path;
    static std::ofstream ofs(FILE_PATH_ + "/save.txt");
    for(auto it = reference_data_.begin(); it != reference_data_.end(); it++){
        ofs << it->equ_file_path << ","
            << it->rgb_file_path << ","
            << it->x << "," << it->y << "," << it->theta << std::endl;
    }
    ofs.close();
}

void ReferenceDataCreator::equ_image_callback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& ex){
        ROS_ERROR("Could not convert to color image");
        return;
    }

    equ_img_ = cv::Mat(cv_ptr->image.rows,cv_ptr->image.cols,cv_ptr->image.type());
    equ_img_ = cv_ptr->image;
    inpaintor_->inpaint_img(cv_ptr->image,equ_img_);
}

void ReferenceDataCreator::rgb_image_callback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& ex){
        ROS_ERROR("Could not convert to color image");
        return;
    }

    rgb_img_ = cv::Mat(cv_ptr->image.rows,cv_ptr->image.cols,cv_ptr->image.type());
    rgb_img_ = cv_ptr->image;
}

void ReferenceDataCreator::pose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
{
    pose_.header = msg->header;
    pose_.pose = msg->pose.pose;
}

void ReferenceDataCreator::create_reference_data()
{
    if(equ_img_.empty() || rgb_img_.empty()) return;
    // if(rgb_img_.empty()) return;

    if(count_%10 == 0){
        std::string equ_file_name = "/equ/image" + std::to_string(count_/10) + ".jpg";
        std::string rgb_file_name = "/rgb/image" + std::to_string(count_/10) + ".jpg";

        cv::imwrite(FILE_PATH_ + equ_file_name,equ_img_);
        cv::imwrite(FILE_PATH_ + rgb_file_name,rgb_img_);

        double x = pose_.pose.position.x;
        double y = pose_.pose.position.y;
        double theta = tf2::getYaw(pose_.pose.orientation);
        reference_data_.push_data(Data(equ_file_name,rgb_file_name,x,y,theta));

        count_++;
    }
    else{
        count_++;
        return;
    }
}

void ReferenceDataCreator::process()
{
    ros::Rate rate(HZ_);
    while(ros::ok()){
        create_reference_data();
        ros::spinOnce();
        rate.sleep();
    }
}

int main(int argc,char** argv)
{
    ros::init(argc,argv,"reference_data_creator");
    ReferenceDataCreator reference_data_creator;
    reference_data_creator.process();
    return 0;
}
