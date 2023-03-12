#include "place_recognition_utils/mismatch_visualizer.h"

using namespace place_recognition;

MismatchVisualizer::MismatchVisualizer() :
    private_nh_("~")
{
    private_nh_.param("HZ",HZ_,{10});
    load_record_file();
    mismatch_poses_pub_ = nh_.advertise<geometry_msgs::PoseArray>("mismatch_poses",1);
}

MismatchVisualizer::~MismatchVisualizer() { }

void MismatchVisualizer::load_record_file()
{
    std::string record_file_path;
    private_nh_.param("RECORD_FILE_PATH",record_file_path,{std::string("")});
    std::string record_file_name = record_file_path + ".txt";
    std::cout << "load: " << record_file_name.c_str() << std::endl;
    std::ifstream ifs(record_file_name);
    std::string line;
    while(std::getline(ifs,line)){
        std::vector<std::string> strvec = split(line,',');
        try{
            double x = static_cast<double>(std::stod(strvec[0]));
            double y = static_cast<double>(std::stod(strvec[1]));
            double theta = static_cast<double>(std::stod(strvec[2]));
            mismatch_points_.emplace_back(MismatchPoint(x,y,theta));
        }
        catch(const std::invalid_argument& ex){
            ROS_ERROR("Invalid: %s", ex.what());
        }
        catch(const std::out_of_range& ex){
            ROS_ERROR("out of range: %s", ex.what());
        }
    }
    ifs.close();

    mismatch_poses_.header.frame_id = "map";
    mismatch_poses_.header.stamp = ros::Time::now();
    for(const auto &p : mismatch_points_){
        geometry_msgs::Pose mismatch_pose;
        mismatch_pose.position.x = p.x;
        mismatch_pose.position.y = p.y;
        mismatch_pose.position.z = 0.0;
        tf2::Quaternion tf_q;
        tf_q.setRPY(0.0,0.0,p.theta);
        mismatch_pose.orientation.w = tf_q.w();
        mismatch_pose.orientation.x = tf_q.x();
        mismatch_pose.orientation.y = tf_q.y();
        mismatch_pose.orientation.z = tf_q.z();
        mismatch_poses_.poses.emplace_back(mismatch_pose);
    }
}

std::vector<std::string> MismatchVisualizer::split(std::string& input,char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while(std::getline(stream,field,delimiter)) result.emplace_back(field);
    return result;
}

void MismatchVisualizer::process()
{
    ros::Rate rate(HZ_);
    while(ros::ok()){
        mismatch_poses_pub_.publish(mismatch_poses_);
        ros::spinOnce();
        rate.sleep();
    }
}

int main(int argc,char** argv)
{
    ros::init(argc,argv,"mismatch_visualizer");
    MismatchVisualizer mismatch_visualizer;
    mismatch_visualizer.process();
    return 0;
}
