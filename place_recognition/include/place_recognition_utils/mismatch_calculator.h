#ifndef MISMATCH_CALCULATOR_H_
#define MISMATCH_CALCULATOR_H_

#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <tf2/utils.h>

#include <sstream>
#include <fstream>

#include "place_recognition_utils/mismatch_point.h"

namespace place_recognition
{
class MismatchCalculator
{
public:
    MismatchCalculator();
    ~MismatchCalculator();
    void process();

private:
    void load_file(std::string file_name);
    std::vector<std::string> split(std::string& input,char delimiter);

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // publisher
    ros::Publisher mismatch_poses_pub_;

    // buffer
    geometry_msgs::PoseArray mismatch_poses_;
    std::vector<MismatchPoint> mismatch_points_;
    int count_;
    int match_count_;
    double rmse_;
    double score_ave_;

    // params
    int HZ_;
    double ERROR_TH_;
};
} // namespace place_recognition

#endif  // MISMATCH_CALCULATOR_H_
