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
        private_nh_("~"),
		file_name_(std::string(""))
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
		create_databse(features);
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
    	
		// save the vocabulary to disk
		std::cout << std::endl << "Saving vocabulary..." << std::endl;
		file_name_ = DIR_PATH_ + get_date() + ".yml.gz";
		vocabulary.save(file_name_);
    	std::cout << "Done" << std::endl;
    }

	// TO DO
	void create_databse(std::vector<cv::Mat>& features)
	{
		std::cout << "Creating a small database..." << std::endl;
		
		// load the vocabulary from disk
		Vocabulary vocabulary(file_name_);
	
	/*
    Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;
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

	std::string get_date()
    {
        time_t t = time(nullptr);
        const tm* localTime = localtime(&t);
        std::stringstream s;
        s << localTime->tm_year + 1900;
        s << std::setw(2) << std::setfill('0') << localTime->tm_mon + 1;
        s << std::setw(2) << std::setfill('0') << localTime->tm_mday;
        s << std::setw(2) << std::setfill('0') << localTime->tm_hour;
        s << std::setw(2) << std::setfill('0') << localTime->tm_min;
        s << std::setw(2) << std::setfill('0') << localTime->tm_sec;

        return s.str();
    }

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // detector
    cv::Ptr<cv::Feature2D> detector_;

	// buffer
	std::string file_name_;

    // params
    std::string DIR_PATH_;
};
}

#endif  // BOW_H_
