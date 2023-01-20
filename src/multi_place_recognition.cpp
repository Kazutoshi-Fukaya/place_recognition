#include "multi_place_recognition/multi_place_recognition.h"

using namespace dbow3;
using namespace place_recognition;

MultiPlaceRecognition::MultiPlaceRecognition() : private_nh_("~")
{
    // detector_mode
    std::string detector_mode;
    private_nh_.param("DETECTOR_MODE",detector_mode,{std::string("orb")});
    set_detector_mode(detector_mode);

    // reference images
    std::string reference_images_path, image_mode;
    private_nh_.param("REFERENCE_IMAGES_PATH",reference_images_path,{std::string("")});
    private_nh_.param("IMAGE_MODE",image_mode,{std::string("rgb")});
    load_reference_images(reference_images_path,image_mode);
    create_database(reference_images_path,image_mode);

    // interface
    private_nh_.param("IS_RECORD",IS_RECORD_,{false});
    interfaces_ = new PlaceRecognitionInterfaces(nh_,private_nh_,database_,image_mode,IS_RECORD_);
    interfaces_->set_reference_image_path(reference_images_path);
    interfaces_->set_detector(detector_);
    interfaces_->set_reference_images(reference_images_);
}

MultiPlaceRecognition::~MultiPlaceRecognition()
{
    if(IS_RECORD_){
        std::string record_path;
        private_nh_.param("RECORD_PATH",record_path,{std::string("")});
        for(size_t i = 0; i < interfaces_->size(); i++){
            std::string file_name = record_path + "pr_match_" + std::to_string(i+1) + ".csv";
            std::ofstream ofs(file_name);
            int match_count = 0;
            for(size_t j = 0; j < interfaces_->at(i)->match_recorder_->size(); j++){
                double error = interfaces_->at(i)->match_recorder_->at(j).get_error();
                if(error > 1.5) match_count++;
                ofs << interfaces_->at(i)->match_recorder_->at(j).time << ","
                    << interfaces_->at(i)->match_recorder_->at(j).est_position.x << ","
                    << interfaces_->at(i)->match_recorder_->at(j).est_position.y << ","
                    << interfaces_->at(i)->match_recorder_->at(j).est_position.theta << ","
                    << interfaces_->at(i)->match_recorder_->at(j).ref_position.x << ","
                    << interfaces_->at(i)->match_recorder_->at(j).ref_position.y << ","
                    << interfaces_->at(i)->match_recorder_->at(j).ref_position.theta << ","
                    << error << std::endl;
            }
            std::cout << "Match Score[" << i << "]: " << (double)match_count/(double)interfaces_->at(i)->match_recorder_->size() << std::endl;
            ofs.close();
        }
    }
}

void MultiPlaceRecognition::set_detector_mode(std::string detector_mode)
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

void MultiPlaceRecognition::load_reference_images(std::string reference_images_path,std::string image_mode)
{
    // load csv file
    std::cout << "=== Load Reference Images ===" << std::endl;
    std::string file_name = reference_images_path + "save.txt";
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
            cv::Mat rgb_image, equ_image;
            if(image_mode == "rgb"){
                rgb_image = cv::imread(reference_images_path + rgb_name,0);
                if(rgb_image.empty()) break;
                std::cout << "file_name: " << rgb_name << std::endl;
                Image rgb;
                calc_features(rgb,rgb_name,rgb_image);
                images.set_rgb_image(rgb);
            }
            else if(image_mode == "equ"){
                equ_image = cv::imread(reference_images_path + equ_name,0);
                if(equ_image.empty()) break;
                std::cout << "file name: " << equ_name << std::endl;
                Image equ;
                calc_features(equ,equ_name,equ_image);
                images.set_equ_image(equ);
            }
            else{
                ROS_ERROR("No applicable 'image_mode'. Please select 'rgb' or 'equ'");
                return;
            }
            reference_images_.emplace_back(images);
        }
        catch(const std::invalid_argument& ex){
            ROS_ERROR("Invalid: %s", ex.what());
        }
        catch(const std::out_of_range& ex){
            ROS_ERROR("out of range: %s", ex.what());
        }
    }
    ifs.close();
    std::cout << std::endl;
}

void MultiPlaceRecognition::calc_features(Image& image,std::string name,cv::Mat img)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(img,cv::Mat(),keypoints,descriptors);
    image.set_params(name,img,keypoints,descriptors);
}

void MultiPlaceRecognition::create_database(std::string reference_images_path,std::string image_mode)
{
    std::cout << "=== Load Database ===" << std::endl;
    std::string database_file = reference_images_path + image_mode + "/dkan_voc.yml.gz";
    std::cout << "load file: " << database_file << std::endl;
    Vocabulary voc(database_file);
    database_ = new Database(voc,false,0);

    if(image_mode == "rgb"){
        for(const auto &img : reference_images_) database_->add(img.rgb.descriptor);
    }
    else if(image_mode == "equ"){
        for(const auto &img : reference_images_) database_->add(img.equ.descriptor);
    }
    else{
        ROS_ERROR("No applicable 'image_mode'. Please select 'rgb' or 'equ'");
        return;
    }

    // info
    database_->get_info();
}

std::vector<std::string> MultiPlaceRecognition::split(std::string& input,char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while(std::getline(stream,field,delimiter)) result.emplace_back(field);
    return result;
}

void MultiPlaceRecognition::process() { ros::spin(); }

int main(int argc,char** argv)
{
    ros::init(argc,argv,"multi_place_recognition");
    MultiPlaceRecognition multi_place_recognition;
    multi_place_recognition.process();
    return 0;
}
