#include "dom_place_recognition/dom_place_recognition_interfaces.h"

using namespace dbow3;
using namespace place_recognition;

DomPlaceRecognitionInterfaces::DomPlaceRecognitionInterfaces() 
{
    ROS_WARN("'dom_place_recognition_interfaces' has not received 'nh', 'private_nh' and 'database'");
}

DomPlaceRecognitionInterfaces::DomPlaceRecognitionInterfaces(ros::NodeHandle nh,ros::NodeHandle private_nh,dbow3::Database* database)
{
    init(nh,private_nh,database,"rgb",false,false);
}

DomPlaceRecognitionInterfaces::DomPlaceRecognitionInterfaces(ros::NodeHandle nh,ros::NodeHandle private_nh,dbow3::Database* database,std::string image_mode,bool is_record,bool is_vis)
{
	init(nh,private_nh,database,image_mode,is_record,is_vis);
}

void DomPlaceRecognitionInterfaces::doms_callback(const dom_estimator_msgs::DomsConstPtr& msg)
{
    if(msg->doms.empty()) return;
    std::vector<DomParam> dom_params;
    for(const auto &dom : msg->doms){
        DomParam dom_param(dom.name,dom.dom,dom.object_size,dom.appearance_count,dom.disappearance_count,dom.observations_count);
        dom_params.emplace_back(dom_param);   
    }

    for(size_t i = 0; i < this->size(); i++){
		this->at(i)->set_dom_params(dom_params);
    }
}

void DomPlaceRecognitionInterfaces::init(ros::NodeHandle nh,ros::NodeHandle private_nh,dbow3::Database* database,std::string image_mode,bool is_record, bool is_vis)
{
    // subscriber
    doms_sub_ = nh.subscribe("dom",1,&DomPlaceRecognitionInterfaces::doms_callback,this);   

    // load params
	this->clear();
	std::string robot_list_name;
	private_nh.param("ROBOT_LIST",robot_list_name,{std::string("robot_list")});
	XmlRpc::XmlRpcValue robot_list;
    if(!private_nh.getParam(robot_list_name.c_str(),robot_list)){
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
            this->at(i) = new DomPlaceRecognitionInterface(nh,name,database,image_mode,is_record,is_vis);
        }
    }
}

void DomPlaceRecognitionInterfaces::set_reference_image_path(std::string path)
{
	for(size_t i = 0; i < this->size(); i++){
		this->at(i)->set_reference_image_path(path);
    }
}

void DomPlaceRecognitionInterfaces::set_detector(cv::Ptr<cv::Feature2D> detector)
{
	for(size_t i = 0; i < this->size(); i++){
		this->at(i)->set_detector(detector);
    }
}

void DomPlaceRecognitionInterfaces::set_reference_images(std::vector<Images> reference_images)
{
	for(size_t i = 0; i < this->size(); i++){
		this->at(i)->set_reference_images(reference_images);
    }
}

void DomPlaceRecognitionInterfaces::update_params(std::vector<Images> updated_reference_images,dbow3::Database* updated_database)
{
    for(size_t i = 0; i < this->size(); i++){
        this->at(i)->update_param(updated_reference_images,updated_database);
    }
}

bool DomPlaceRecognitionInterfaces::stocked_enough_images(std::vector<Images>& collected_images)
{
    int count = 0;
    std::vector<Images> collected_images_;
    for(size_t i = 0; i < this->size(); i++){
        std::vector<Images> stored_images = this->at(i)->get_stored_images();
        for(const auto &si : stored_images){
            Image image = si.rgb;
            image.file_name = std::to_string(count);
            Images images(si.x,si.y,si.theta);
            images.set_rgb_image(image);
            collected_images_.emplace_back(images);
            count++;
        }
    }
    collected_images_ = collected_images;
    if(collected_images_.size() > 30) return true;
    else return false;
}