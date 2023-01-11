#include "place_recognition/batch_place_recognition.h"

using namespace dbow3;
using namespace place_recognition;

BatchPlaceRecognition::BatchPlaceRecognition() :
	private_nh_("~"),
	detector_(cv::ORB::create())
{
	private_nh_.param("REFERENCE_FILE_PATH",REFERENCE_FILE_PATH_,{std::string("")});
	private_nh_.param("QUERY_FILE_PATH",QUERY_FILE_PATH_,{std::string("")});  
	private_nh_.param("MODE",MODE_,{std::string("rgb")});

	std::cout << QUERY_FILE_PATH_ << std::endl;

    load_reference_images();
    create_database();
}

void BatchPlaceRecognition::load_reference_images()
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

            cv::Mat equ_image = cv::imread(REFERENCE_FILE_PATH_ + equ_name,0);
            cv::Mat rgb_image = cv::imread(REFERENCE_FILE_PATH_ + rgb_name,0);
            if(equ_image.empty() || rgb_image.empty()) break;
            std::cout << "file name: " << REFERENCE_FILE_PATH_ + equ_name << std::endl;
            std::cout << "file_name: " << REFERENCE_FILE_PATH_ + rgb_name << std::endl;

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

void BatchPlaceRecognition::calc_features(Image& image,std::string name,cv::Mat img)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(img,cv::Mat(),keypoints,descriptors);
    image.set_params(name,img,keypoints,descriptors);
}

void BatchPlaceRecognition::create_database()
{
    std::cout << "=== Load Database ===" << std::endl;
	std::string file_name = REFERENCE_FILE_PATH_ + MODE_ + "/dkan_mono.yml.gz";
    std::cout << "load file: " << file_name << std::endl;
    Vocabulary voc(file_name);
    Database tmp_db(voc,false,0);
    db_ = tmp_db;

    // rgb
	if(MODE_ == std::string("rgb")){
    	for(const auto &img : reference_images_) db_.add(img.rgb.descriptor);
	}
	else if(MODE_ == std::string("equ")){
		for(const auto &img : reference_images_) db_.add(img.equ.descriptor);
	}

    // info
    db_.get_info();
}

std::vector<std::string> BatchPlaceRecognition::split(std::string& input,char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while(std::getline(stream,field,delimiter)) result.emplace_back(field);
    return result;
}

void BatchPlaceRecognition::process()
{
	std::cout << "=== Main Process ===" << std::endl;

	std::cout << "=== Load Query Images ===" << std::endl;
    std::string file_name = QUERY_FILE_PATH_ + "/save.txt";
    std::cout << "load: " << file_name << std::endl;
	static std::ofstream ofs(QUERY_FILE_PATH_ + "/" + MODE_ + "_result.txt");
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

            Image equ, rgb;				// image
            Images images(x,y,theta);	// images
			QueryResults ret;
			if(MODE_ == std::string("rgb")){
				cv::Mat rgb_image = cv::imread(QUERY_FILE_PATH_ + rgb_name,0);
				if(rgb_image.empty()) break;
				std::cout << "file_name: " << QUERY_FILE_PATH_ + rgb_name << std::endl;
				calc_features(rgb,rgb_name,rgb_image);
				images.set_rgb_image(rgb);
				db_.query(images.rgb.descriptor,ret,4);
			}
			else if(MODE_ == std::string("equ")){
				cv::Mat equ_image = cv::imread(QUERY_FILE_PATH_ + equ_name,0);
            	if(equ_image.empty()) break;
            	std::cout << "file name: " << QUERY_FILE_PATH_ + equ_name << std::endl;
				calc_features(equ,equ_name,equ_image);
				images.set_equ_image(equ);
				db_.query(images.equ.descriptor,ret,4);
			}

			std::cout << ret << std::endl << std::endl;
    		if(ret.empty()) continue;

			// output file
			if(MODE_ == std::string("rgb")) ofs << QUERY_FILE_PATH_ + rgb_name << ",";
			else if(MODE_ == std::string("equ")) ofs << QUERY_FILE_PATH_ + equ_name << ",";
			ofs << images.x << "," << images.y << "," << images.theta << ",";
			for(int i = 0; i < 4; i++){
				std::string name = REFERENCE_FILE_PATH_ + MODE_ + "/image" + std::to_string(ret[i].id) + ".jpg";
				ofs << name << ","
				    << reference_images_.at(ret[i].id).x << ","
				    << reference_images_.at(ret[i].id).y << ","
					<< reference_images_.at(ret[i].id).theta << ",";
					if(i == 3) ofs << ret[i].score << std::endl;
					else ofs << ret[i].score << ",";
			}
        }
        catch(const std::invalid_argument& ex){
            std::cerr << "Invalid: " << ex.what() << std::endl;
        }
        catch(const std::out_of_range& ex){
            ROS_ERROR("out of range: %s", ex.what());
        }
    }
    ifs.close();
	ofs.close();
    std::cout << std::endl;
}

int main(int argc,char** argv)
{
	ros::init(argc,argv,"batch_place_recognition");
	BatchPlaceRecognition batch_place_recognition;
	batch_place_recognition.process();
	return 0;
}