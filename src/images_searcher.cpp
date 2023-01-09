#include <ros/ros.h>

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
		descriptor(_descriptor)
	{}

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

class ImagesSearcher
{
public:
	ImagesSearcher();
	void process();

private:
	bool load_query_image();
	void load_reference_images();
	void calc_features(Image& image,std::string name,cv::Mat img);

	void create_database();

	std::vector<std::string> split(std::string& input,char delimiter);

	// node handler
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;

	// detector
    cv::Ptr<cv::Feature2D> detector_;

	// Query
	Image query_image_;

	// Reference
	std::vector<Images> reference_images_;

	// buffer
	std::string file_path_;

	// param
	std::string DIR_PATH_;
	std::string REFERENCE_FILE_PATH_;
};
}

using namespace dbow3;
using namespace place_recognition;

ImagesSearcher::ImagesSearcher() :
	private_nh_("~"),
	detector_(cv::ORB::create()),
	file_path_(std::string(""))
{
	private_nh_.param("DIR_PATH",DIR_PATH_,{std::string("")});

    private_nh_.param("REFERENCE_FILE_NAME",REFERENCE_FILE_PATH_,{std::string("")});
    // file_path_ = REFERENCE_FILE_PATH_ + "rgb/dkan_mono.yml.gz";
	file_path_ = REFERENCE_FILE_PATH_ + "equ/dkan_mono.yml.gz";
}

bool ImagesSearcher::load_query_image()
{
	std::cout << "=== Load Query Image ===" << std::endl;
	// std::string file_path = DIR_PATH_ + "rgb/image18.jpg";
	std::string file_path = DIR_PATH_ + "equ/image8.jpg";
	cv::Mat image = cv::imread(file_path,0);
	if(image.empty()){
		std::cerr << "No query image" << std::endl;
		return false;
	}
	std::cout << "file name: " << file_path << std::endl << std::endl;
	std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
	detector_->detectAndCompute(image,cv::Mat(),keypoints,descriptors);
	query_image_.set_params(file_path,image,keypoints,descriptors);
	return true;
}

void ImagesSearcher::load_reference_images()
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
			
			cv::Mat equ_image = cv::imread(equ_name,0);
			cv::Mat rgb_image = cv::imread(rgb_name,0);
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

void ImagesSearcher::calc_features(Image& image,std::string name,cv::Mat img)
{
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	detector_->detectAndCompute(img,cv::Mat(),keypoints,descriptors);
	image.set_params(name,img,keypoints,descriptors);
}

void ImagesSearcher::create_database()
{
	std::cout << "=== Load Database ===" << std::endl;
	std::cout << "load file: " << file_path_ << std::endl;
	Vocabulary voc(file_path_);
	Database db(voc,false,0);

	// rgb
	for(const auto &img : reference_images_) db.add(img.rgb.descriptor);

	// info
	db.get_info();

	QueryResults ret;
	db.query(query_image_.descriptor,ret,4);
	// std::cout << "Searching for Image " << DIR_PATH_ + "rgb/image18.jpg" << ". " << ret << std::endl;
	std::cout << "Searching for Image " << DIR_PATH_ + "equ/image8.jpg" << ". " << ret << std::endl;
}

std::vector<std::string> ImagesSearcher::split(std::string& input,char delimiter)
{
	std::istringstream stream(input);
	std::string field;
	std::vector<std::string> result;
	while(std::getline(stream,field,delimiter)) result.emplace_back(field);
    return result;
}

void ImagesSearcher::process()
{
	if(!load_query_image()) return;
	load_reference_images();

	create_database();
}

int main(int argc,char** argv)
{
	ros::init(argc,argv,"images_searcher");
	ImagesSearcher images_searcher;
	images_searcher.process();
	return 0;
}