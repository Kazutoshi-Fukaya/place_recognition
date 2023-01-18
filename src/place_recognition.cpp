#include "place_recognition/place_recognition.h"

using namespace dbow3;
using namespace place_recognition;

PlaceRecognition::PlaceRecognition() : private_nh_("~")
{
    private_nh_.param("REFERENCE_FILE_NAME",REFERENCE_FILE_PATH_,{std::string("")});
    private_nh_.param("IMAGE_MODE",IMAGE_MODE_,{std::string("rgb")});

    std::string detector_mode;
    private_nh_.param("DETECTOR_MODE",detector_mode,{std::string("orb")});
    set_detector_mode(detector_mode);

    load_reference_images();
    create_database();

    img_sub_ = nh_.subscribe("img_in",1,&PlaceRecognition::image_callback,this);
    img_pub_ = nh_.advertise<sensor_msgs::Image>("img_out",1);
    
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pose_out",1);
}

void PlaceRecognition::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& ex){
        ROS_ERROR("Could not convert to color image");
        return;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(cv_ptr->image,cv::Mat(),keypoints,descriptors);

    QueryResults ret;
    database_->query(descriptors,ret,4);
    // std::cout << ret << std::endl << std::endl;
    if(ret.empty()) return;

    int id = ret.at(0).id;
    std::string name = REFERENCE_FILE_PATH_ + "rgb/image" + std::to_string(id) + ".jpg";
    cv::Mat output_img = cv::imread(name);
    if(output_img.empty()) return;

    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8",output_img).toImageMsg();
    img_pub_.publish(img_msg);

    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.header.stamp = ros::Time::now();
    pose.pose.position.x = reference_images_.at(id).x;
    pose.pose.position.y = reference_images_.at(id).y;
    pose.pose.position.z = 0;
    tf2::Quaternion tf_q;
    tf_q.setRPY(0.0,0.0,reference_images_.at(id).theta);
    pose.pose.orientation.w = tf_q.w();
    pose.pose.orientation.x = tf_q.x();
    pose.pose.orientation.y = tf_q.y();
    pose.pose.orientation.z = tf_q.z();

    pose_pub_.publish(pose);
}

void PlaceRecognition::set_detector_mode(std::string detector_mode)
{
    if(detector_mode == "orb") detector_ = cv::ORB::create();
    else if(detector_mode == "brisk") detector_ = cv::BRISK::create();
    else if(detector_mode == "akaze") detector_ = cv::AKAZE::create();
    else{
		ROS_WARN("No applicable 'detector_mode'. Please select 'orb', 'brisk' or 'akaze'");
		ROS_INFO("Set 'orb'");
        detector_ = cv::ORB::create();
    }
}

void PlaceRecognition::load_reference_images()
{
    // load csv file
    std::cout << "=== Load Reference Images ===" << std::endl;
    std::string file_name = REFERENCE_FILE_PATH_ + "save.txt";
    std::cout << "load: " << file_name << std::endl;
    std::ifstream ifs(file_name);
    std::string line;
    while(std::getline(ifs,line)){
        std::vector<std::string> strvec = split(line,',');
        try{
            std::string equ_name = static_cast<std::string>(strvec[0]);
            std::string rgb_name = static_cast<std::string>(strvec[1]);
            double x = static_cast<double>(std::stod(strvec[2]));
            double y = static_cast<double>(std::stod(strvec[3]));
            double theta = static_cast<double>(std::stod(strvec[4]));

            Images images(x,y,theta);
            if(IMAGE_MODE_ == "rgb"){
                cv::Mat rgb_image = cv::imread(REFERENCE_FILE_PATH_ + rgb_name,0);
                if(rgb_image.empty()) break;
                // std::cout << "file_name: " << rgb_name << std::endl;

                Image rgb;
                calc_features(rgb,rgb_name,rgb_image);
                images.set_rgb_image(rgb);
            }
            else if(IMAGE_MODE_ == "equ"){
                cv::Mat equ_image = cv::imread(REFERENCE_FILE_PATH_ + equ_name,0);
                if(equ_image.empty()) break;
                // std::cout << "file name: " << equ_name << std::endl;
                
                Image equ;
                calc_features(equ,equ_name,equ_image);
                images.set_equ_image(equ);
            }
            else{
                ROS_ERROR("Invalid Mode");
                ROS_INFO("Please set 'rgb' or 'equ'");
            }

            reference_images_.emplace_back(images);
        }
        catch(const std::invalid_argument& ex){
            std::cerr << "Invalid: " << ex.what() << std::endl;
        }
        catch(const std::out_of_range& ex){
            ROS_ERROR("out of range: %s", ex.what());
        }
    }
    ifs.close();
    std::cout << std::endl;
}

void PlaceRecognition::calc_features(Image& image,std::string name,cv::Mat img)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(img,cv::Mat(),keypoints,descriptors);
    image.set_params(name,img,keypoints,descriptors);
}

void PlaceRecognition::create_database()
{
    std::cout << "=== Load Database ===" << std::endl;
    std::string file_path_ = REFERENCE_FILE_PATH_ + IMAGE_MODE_ + "/dkan_mono.yml.gz";
    std::cout << "load file: " << file_path_ << std::endl;
    Vocabulary voc(file_path_);
    database_ = new Database(voc,false,0);

    // add
    for(const auto &img : reference_images_){
        if(IMAGE_MODE_ == "rgb") database_->add(img.rgb.descriptor);
        else if(IMAGE_MODE_ == "equ") database_->add(img.equ.descriptor);
    }

    // info
    database_->get_info();
}

std::vector<std::string> PlaceRecognition::split(std::string& input,char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while(std::getline(stream,field,delimiter)) result.emplace_back(field);
    return result;
}

void PlaceRecognition::process() { ros::spin(); }

int main(int argc,char** argv)
{
    ros::init(argc,argv,"place_recognition");
    PlaceRecognition place_recognition;
    place_recognition.process();
    return 0;
}