#ifndef IMAGE_H_
#define IMAGE_H_

#include <iostream>
#include <opencv2/opencv.hpp>

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
        descriptor(_descriptor) {}

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
} // namespace place_recognition

#endif	// IMAGE_H_