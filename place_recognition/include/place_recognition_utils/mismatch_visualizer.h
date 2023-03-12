#ifndef MISMATCH_VISUALIZER_H_
#define MISMATCH_VISUALIZER_H_

#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <tf2/utils.h>

#include <sstream>
#include <fstream>

#include "place_recognition_utils/mismatch_point.h"

namespace place_recognition
{
class MismatchVisualizer
{
public:
    MismatchVisualizer();
    ~MismatchVisualizer();
    void process();

private:
    void load_record_file();
    std::vector<std::string> split(std::string& input,char delimiter);

    // node handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // publisher
    ros::Publisher mismatch_poses_pub_;

    // buffer
    geometry_msgs::PoseArray mismatch_poses_;
    std::vector<MismatchPoint> mismatch_points_;

    // params
    int HZ_;
};
} // namespace place_recognition

#endif  // MISMATCH_VISUALIZER_H_
