#ifndef BOW_H_
#define BOW_H_

// ros
#include <ros/ros.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "bow/vocabulary/vocabulary.h"

namespace place_recognition
{
class BoW
{
public:
    BoW() :
        private_nh_("~")
    {
        std::string mode;
        private_nh_.param("MODE",mode,{std::string("orb")});
        set_detector_mode(mode);

        private_nh_.param("DIR_PATH",DIR_PATH_,{std::string("")});
    }

    void process()
    {
        std::vector<cv::Mat> features = load_features();
        create_vocabulary(features);
    }

private:
    void set_detector_mode(std::string mode)
    {
        if(mode == "orb") detector_ = cv::ORB::create();
        else if(mode == "brisk") detector_ = cv::BRISK::create();
        else if(mode == "akaze") detector_ = cv::AKAZE::create();
        else{
            std::cerr << "No applicable mode. Please select 'orb', 'brisk' or 'akaze'" << std::endl;
            std::cerr  << "Set 'orb" << std::endl;
            detector_ = cv::ORB::create();
        }
    }

    void create_vocabulary(std::vector<cv::Mat>& features)
    {
        const int k = 9;
        const int L = 3;
        const WeightingType weighting_type = TF_IDF;
        const ScoringType scoring_type = L1_NORM;
        Vocabulary vocabulary(k,L,weighting_type,scoring_type);

        std::cout << "Creating small " << k << "^" << L << " vocabulary ..." << std::endl;
        vocabulary.create(features);
		std::cout << "... done" << std::endl;
		
		std::cout << "Vocabulary information: " << std::endl;
        vocabulary.get_info();

		// lets do something with this vocabulary
		std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;
		BowVector v1, v2;
		for(size_t i = 0; i < features.size(); i++){
			vocabulary.transform(features[i],v1);
			for(size_t j = 0; j < features.size(); j++){
				vocabulary.transform(features[j],v2);
            	double score = vocabulary.score(v1,v2);
            	std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
        	}
    	}
	/*
    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
	*/
    }

    std::vector<cv::Mat> load_features()
    {
        std::cout << "Extracting features ..." << std::endl;
        std::vector<cv::Mat> features;
        for(size_t i = 0; ; i++){
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            std::string file_path = DIR_PATH_ + std::to_string(i) + "_resized.jpg";
            std::cout << "file name: " << file_path << std::endl;
            cv::Mat image = cv::imread(file_path,0);
            if(image.empty()) break;
            detector_->detectAndCompute(image,cv::Mat(),keypoints,descriptors);
            features.emplace_back(descriptors);
        }
        std::cout << std::endl;
        return features;
    }

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // detector
    cv::Ptr<cv::Feature2D> detector_;

    // params
    std::string DIR_PATH_;
};
}

#endif  // BOW_H_
