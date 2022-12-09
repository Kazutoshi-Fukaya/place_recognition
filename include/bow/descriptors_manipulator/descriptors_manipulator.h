#ifndef DESCRIPTORS_MANIPULATOR_H_
#define DESCRIPTORS_MANIPULATOR_H_

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

namespace place_recognition
{
class DescriptorsManipulator
{
public:
	void mean_value(std::vector<cv::Mat>& descriptors,cv::Mat &mean);
    double distance(cv::Mat& a,cv::Mat& b);
    uint32_t distance_8uc1(cv::Mat& a,cv::Mat& b);
	size_t get_desc_size_bytes(cv::Mat& d);
};
}

#endif	// DESCRIPTORS_MANIPULATOR_H_
