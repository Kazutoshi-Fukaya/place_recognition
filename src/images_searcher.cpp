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

class ImagesSearcher
{
public:
	ImagesSearcher();
	void process();

private:
	bool load_query_image();
	void load_reference_images();
	void create_database();

	// node handler
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;

	// detector
    cv::Ptr<cv::Feature2D> detector_;

	// Query
	Image query_image_;

	// Reference
	std::vector<Image> reference_images_;

	// buffer
	std::string file_path_;

	// param
	std::string DIR_PATH_;
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
    file_path_ = DIR_PATH_ + "dkan_mono.yml.gz";
}

bool ImagesSearcher::load_query_image()
{
	std::cout << "=== Load Query Image ===" << std::endl;
	std::string file_path = DIR_PATH_ + "/rgb/image10.jpg";
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
	std::cout << "Load Reference Images ===" << std::endl;
	for(size_t i = 0; ; i++){
		std::string file_path = DIR_PATH_ + "/rgb/image" + std::to_string(i) + ".jpg";
		cv::Mat image = cv::imread(file_path,0);
		if(image.empty()) break;
		std::cout << "file name: " << file_path << std::endl;
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		detector_->detectAndCompute(image,cv::Mat(),keypoints,descriptors);
		reference_images_.emplace_back(Image(file_path,image,keypoints,descriptors));
	}
	std::cout << std::endl;
}

void ImagesSearcher::create_database()
{
	std::cout << "=== Load Database ===" << std::endl;
	std::cout << "load file: " << file_path_ << std::endl;
	Vocabulary voc(file_path_);
	Database db(voc,false,0);

	for(const auto &img : reference_images_) db.add(img.descriptor);

	// info
	db.get_info();

	QueryResults ret;
	db.query(query_image_.descriptor,ret,4);
	std::cout << "Searching for Image " << DIR_PATH_ + "/rgb/image10.jpg" << ". " << ret << std::endl;
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