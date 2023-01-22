#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/utils.h>

#include "dom_estimator_msgs/Doms.h"
#include "object_detector_msgs/ObjectPositions.h"

#include <random>

#include "dbow3/vocabulary/vocabulary.h"
#include "dbow3/database/database.h"

namespace place_recognition
{
class Image
{
public:
    Image() {}

    Image(std::string _file_name,cv::Mat _img,
          std::vector<cv::KeyPoint> _keypoints,
          cv::Mat _descriptor) :
        file_name(_file_name),
        img(_img),
        keypoints(_keypoints),
        descriptor(_descriptor){}

    void set_params(std::string _file_name,cv::Mat _img,
                    std::vector<cv::KeyPoint> _keypoints,
                    cv::Mat _descriptor)
    {
        file_name = _file_name;
        img = _img;
        keypoints = _keypoints;
        descriptor = _descriptor;
    }

    std::string file_name;
    cv::Mat img;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;

private:
};

class Images
{
public:
    Images() :
        x(0.0), y(0.0), theta(0.0) {}

    Images(double _x,double _y,double _theta) :
        x(_x), y(_y), theta(_theta) {}

    void set_equ_image(Image _equ) { equ = _equ; }
    void set_rgb_image(Image _rgb) { rgb = _rgb; }

    Image equ;
    Image rgb;

    double x;
    double y;
    double theta;

private:
};

class ObjectTime
{
public:
    ObjectTime() {}
    ObjectTime(std::string _name) :
        name(_name) {}

    std::string name;
    ros::Time last_time;
    double time;
private:
};

class PlaceRecognition
{
public:
    PlaceRecognition();
    void process();

private:
    void image_callback(const sensor_msgs::ImageConstPtr& msg);
    void dom_callback(const dom_estimator_msgs::DomsConstPtr& msg);
    void obj_callback(const object_detector_msgs::ObjectPositionsConstPtr& msg);

    void load_reference_images();
    void calc_features(Image& image,std::string name,cv::Mat img);

    void create_database();

    std::vector<std::string> split(std::string& input,char delimiter);

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // subscriber
    ros::Subscriber img_sub_;
    ros::Subscriber dom_sub_;
    ros::Subscriber obj_sub_;

    // publisher
    ros::Publisher img_pub_;
    ros::Publisher pose_pub_;

    // database
    dbow3::Database db;

    // detector
    cv::Ptr<cv::Feature2D> detector_;

    // Query
    // Image query_image_;

    // Reference
    std::vector<Images> reference_images_;

    // random
    std::random_device seed_;
    std::mt19937 engine_;

    // buffer
    dom_estimator_msgs::Doms doms_;
    object_detector_msgs::ObjectPositions obj_;
    geometry_msgs::PoseStamped last_pose_;
    std::vector<ObjectTime> objects_time_;
    std::string file_path_;

    // param
    std::string DIR_PATH_;
    std::string REFERENCE_FILE_PATH_;
    int HZ_;
};
}

using namespace dbow3;
using namespace place_recognition;

PlaceRecognition::PlaceRecognition() :
    private_nh_("~"),
    detector_(cv::ORB::create()),
    engine_(seed_()),
    file_path_(std::string(""))
{
    private_nh_.param("DIR_PATH",DIR_PATH_,{std::string("")});

    private_nh_.param("REFERENCE_FILE_NAME",REFERENCE_FILE_PATH_,{std::string("")});
    file_path_ = REFERENCE_FILE_PATH_ + "rgb/dkan_mono.yml.gz";
    // file_path_ = REFERENCE_FILE_PATH_ + "equ/dkan_mono.yml.gz";

    private_nh_.param("HZ",HZ_,{10});

    load_reference_images();
    create_database();

    img_sub_ = nh_.subscribe("img_in",1,&PlaceRecognition::image_callback,this);
    dom_sub_ = nh_.subscribe("dom",1,&PlaceRecognition::dom_callback,this);
    obj_sub_ = nh_.subscribe("obj_in",1,&PlaceRecognition::obj_callback,this);

    img_pub_ = nh_.advertise<sensor_msgs::Image>("img_out",1);
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pose_out",1);

    // objects time (あとで作りなおす)
    objects_time_.emplace_back(ObjectTime(std::string("trash_can")));
    objects_time_.emplace_back(ObjectTime(std::string("fire_hydrant")));
    objects_time_.emplace_back(ObjectTime(std::string("bench")));
    objects_time_.emplace_back(ObjectTime(std::string("table")));
    objects_time_.emplace_back(ObjectTime(std::string("chair")));
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

    // change image (あとで作りなおす)
    std::uniform_real_distribution<> dist(0.0,1.0);
    for(const auto &dom : doms_.doms){
        if(dom.dom > 0.0){
            for(const auto &obj_time : objects_time_){
                if(dom.name == obj_time.name){
                    double lambda = dom.dom*obj_time.time;
                    double p_0 = std::exp(-1*lambda);
                    if(dist(engine_) < p_0){
                        // rgb image
                        Image rgb;
                        calc_features(rgb,"new_fig",cv_ptr->image);
                        Images images(last_pose_.pose.position.x,last_pose_.pose.position.y,tf2::getYaw(last_pose_.pose.orientation));
                        images.set_rgb_image(rgb);
                        reference_images_.emplace_back(images);
                    }
                }
            }
        }
    }


    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(cv_ptr->image,cv::Mat(),keypoints,descriptors);

    QueryResults ret;
    db.query(descriptors,ret,4);
    std::cout << ret << std::endl;

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

    last_pose_ = pose;
    pose_pub_.publish(pose);
}

void PlaceRecognition::dom_callback(const dom_estimator_msgs::DomsConstPtr& msg) { doms_ = *msg; }

void PlaceRecognition::obj_callback(const object_detector_msgs::ObjectPositionsConstPtr& msg)
{
    obj_ = *msg;
    for(auto obj_time : objects_time_){
        for(const auto &obj : msg->object_position){
            if(obj_time.name == obj.Class){
                obj_time.time = (msg->header.stamp - obj_time.last_time).toSec();
                obj_time.last_time = msg->header.stamp;
            }
        }
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

            cv::Mat equ_image = cv::imread(DIR_PATH_ + equ_name,0);
            cv::Mat rgb_image = cv::imread(DIR_PATH_ + rgb_name,0);
            if(equ_image.empty() || rgb_image.empty()) break;
            std::cout << "file name: " << equ_name << std::endl;
            std::cout << "file_name: " << rgb_name << std::endl;

            // equ image
            Image equ;
            calc_features(equ,equ_name,equ_image);

            // rgb image
            Image rgb;
            calc_features(rgb,rgb_name,rgb_image);

            Images images(x,y,theta);
            images.set_equ_image(equ);
            images.set_rgb_image(rgb);
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
    std::cout << "load file: " << file_path_ << std::endl;
    Vocabulary voc(file_path_);
    Database tmp_db(voc,false,0);
    db = tmp_db;

    // rgb
    for(const auto &img : reference_images_) db.add(img.rgb.descriptor);

    // info
    db.get_info();

    // QueryResults ret;
    // db.query(query_image_.descriptor,ret,4);
    // std::cout << "Searching for Image " << DIR_PATH_ + "rgb/image18.jpg" << ". " << ret << std::endl;
    // std::cout << "Searching for Image " << DIR_PATH_ + "equ/image8.jpg" << ". " << ret << std::endl;
}

std::vector<std::string> PlaceRecognition::split(std::string& input,char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while(std::getline(stream,field,delimiter)) result.emplace_back(field);
    return result;
}

void PlaceRecognition::process()
{
    ros::Rate rate(HZ_);
    while(ros::ok()){
        ros::spinOnce();
        rate.sleep();
    }
}

int main(int argc,char** argv)
{
    ros::init(argc,argv,"place_recognition");
    PlaceRecognition place_recognition;
    place_recognition.process();
    return 0;
}
