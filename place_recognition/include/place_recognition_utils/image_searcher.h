#ifndef IMAGE_SEARCHER_H_
#define IMAGE_SEARCHER_H_

#include <ros/ros.h>

#include "dbow3/vocabulary/vocabulary.h"
#include "dbow3/database/database.h"

#include "place_recognition/images.h"

namespace place_recognition
{
class ImageSearcher
{
public:
    ImageSearcher();
    void process();

private:
    void set_detector_mode(std::string mode);
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

    // param
    std::string QUERY_IMAGE_NAME_;
    std::string REFERENCE_FILE_PATH_;
    std::string IMAGE_MODE_;
};
} // namespace place_recognition

#endif  // IMAGE_SEARCHER_H_
