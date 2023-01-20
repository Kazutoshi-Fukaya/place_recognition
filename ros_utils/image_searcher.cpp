#include "place_recognition_utils/image_searcher.h"

using namespace dbow3;
using namespace place_recognition;

ImageSearcher::ImageSearcher() : private_nh_("~")
{
    // detector_mode
    std::string detector_mode;
    private_nh_.param("DETECTOR_MODE",detector_mode,{std::string("orb")});
    set_detector_mode(detector_mode);

    private_nh_.param("IMAGE_MODE",IMAGE_MODE_,{std::string("rgb")});
    private_nh_.param("QUERY_IMAGE_NAME",QUERY_IMAGE_NAME_,{std::string("")});
    private_nh_.param("REFERENCE_FILE_PATH",REFERENCE_FILE_PATH_,{std::string("")});
}

void ImageSearcher::set_detector_mode(std::string mode)
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

bool ImageSearcher::load_query_image()
{
    std::cout << "=== Load Query Image ===" << std::endl;
    std::cout << "load image: " << QUERY_IMAGE_NAME_ << std::endl << std::endl;
    cv::Mat image = cv::imread(QUERY_IMAGE_NAME_,0);
    if(image.empty()){
        ROS_ERROR("No query image");
        return false;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(image,cv::Mat(),keypoints,descriptors);
    query_image_.set_params(QUERY_IMAGE_NAME_,image,keypoints,descriptors);
    return true;
}

void ImageSearcher::load_reference_images()
{
    // load file
    std::cout << "=== Load Reference Images ===" << std::endl;
    std::string file_name = REFERENCE_FILE_PATH_ + "save.txt";
    std::cout << "load txt: " << file_name << std::endl;
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
                std::string rgb_image_name = REFERENCE_FILE_PATH_ + rgb_name;
                cv::Mat rgb_image = cv::imread(rgb_image_name,0);
                if(rgb_image.empty()) break;
                std::cout << "file_name: " << rgb_image_name << std::endl;
                Image rgb;
                calc_features(rgb,rgb_name,rgb_image);
                images.set_rgb_image(rgb);
            }
            else if(IMAGE_MODE_ == "equ"){
                std::string equ_image_name = REFERENCE_FILE_PATH_ + equ_name;
                cv::Mat equ_image = cv::imread(equ_image_name,0);
                if(equ_image.empty()) break;
                std::cout << "file name: " << equ_image_name << std::endl;
                Image equ;
                calc_features(equ,equ_name,equ_image);
                images.set_equ_image(equ);
            }
            reference_images_.emplace_back(images);
        }
        catch(const std::invalid_argument& ex){
            ROS_ERROR("Invalid: %s ",ex.what());
        }
        catch(const std::out_of_range& ex){
            ROS_ERROR("out of range: %s", ex.what());
        }
    }
    ifs.close();
    std::cout << std::endl;
}

void ImageSearcher::calc_features(Image& image,std::string name,cv::Mat img)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(img,cv::Mat(),keypoints,descriptors);
    image.set_params(name,img,keypoints,descriptors);
}

void ImageSearcher::create_database()
{
    std::cout << "=== Load Database ===" << std::endl;
    std::string file_name = REFERENCE_FILE_PATH_ + IMAGE_MODE_ + "/dkan_voc.yml.gz";
    std::cout << "load file: " << file_name << std::endl;
    Vocabulary voc(file_name);
    Database db(voc,false,0);

    for(const auto &img : reference_images_){
        if(IMAGE_MODE_ == "rgb") db.add(img.rgb.descriptor);
        else if(IMAGE_MODE_ == "equ") db.add(img.equ.descriptor);
    }

    // info
    db.get_info();

    QueryResults ret;
    db.query(query_image_.descriptor,ret,4);
    std::cout << "Searching for Image " << QUERY_IMAGE_NAME_ << std::endl<< ret << std::endl;
}

std::vector<std::string> ImageSearcher::split(std::string& input,char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while(std::getline(stream,field,delimiter)) result.emplace_back(field);
    return result;
}

void ImageSearcher::process()
{
    if(!load_query_image()) return;
    load_reference_images();
    create_database();
}

int main(int argc,char** argv)
{
    ros::init(argc,argv,"images_searcher");
    ImageSearcher image_searcher;
    image_searcher.process();
    return 0;
}
