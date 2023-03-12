#include "place_recognition_utils/mismatch_calculator.h"

using namespace place_recognition;

MismatchCalculator::MismatchCalculator() :
    private_nh_("~"),
    count_(0), match_count_(0),
    rmse_(0.0), score_ave_(0.0)
{
    private_nh_.param("HZ",HZ_,{10});
    private_nh_.param("ERROR_TH",ERROR_TH_,{1.5});

    std::string load_file_name;
    private_nh_.param("LOAD_FILE_NAME",load_file_name,{std::string("")});
    load_file(load_file_name);

    mismatch_poses_pub_ = nh_.advertise<geometry_msgs::PoseArray>("mismatch_poses",1);
}

MismatchCalculator::~MismatchCalculator()
{
    bool is_record_;
    private_nh_.param("IS_RECORD",is_record_,{false});
    if(is_record_){
        std::string record_file_path;
        private_nh_.param("RECORD_FILE_PATH",record_file_path,{std::string("")});
        std::ofstream ofs(record_file_path + ".txt");
        for(const auto &p : mismatch_points_){
            ofs << p.x << "," << p.y << "," << p.theta << std::endl;
        }
        ofs.close();
    }
}

void MismatchCalculator::load_file(std::string file_name)
{
    double error = 0.0;
    std::cout << "load: " << file_name << std::endl;
    std::ifstream ifs(file_name);
    std::string line;
    while(std::getline(ifs,line)){
        std::vector<std::string> strvec = split(line,',');
        try{
            double x_hat = static_cast<double>(std::stod(strvec[1]));
            double y_hat = static_cast<double>(std::stod(strvec[2]));
            double theta_hat = static_cast<double>(std::stod(strvec[3]));
            double x_pr = static_cast<double>(std::stod(strvec[5]));
            double y_pr = static_cast<double>(std::stod(strvec[6]));
            double score = static_cast<double>(std::stod(strvec[8]));

            double diff_x = x_hat - x_pr;
            double diff_y = y_hat - y_pr;
            if(std::sqrt(diff_x*diff_x + diff_y*diff_y) < ERROR_TH_) match_count_++;
            else mismatch_points_.emplace_back(MismatchPoint(x_hat,y_hat,theta_hat));

            error += diff_x*diff_y;
            error += diff_y*diff_y;
            score_ave_ += score;
            count_++;
        }
        catch(const std::invalid_argument& ex){
            ROS_ERROR("Invalid: %s", ex.what());
        }
        catch(const std::out_of_range& ex){
            ROS_ERROR("out of range: %s", ex.what());
        }
    }
    ifs.close();
    rmse_ = std::sqrt(error/(double)count_);
    score_ave_ /= (double)count_;

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

std::vector<std::string> MismatchCalculator::split(std::string& input,char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while(std::getline(stream,field,delimiter)) result.emplace_back(field);
    return result;
}

void MismatchCalculator::process()
{
    std::cout << "===== RESULT ====" << std::endl;
    std::cout << "RMSE: " << rmse_ << std::endl;
    std::cout << "Score Average: " << score_ave_ << std::endl;
    std::cout << "Matching Rate: " << (double)match_count_/(double)count_ << std::endl;
    ros::Rate rate(HZ_);
    while(ros::ok()){
        mismatch_poses_pub_.publish(mismatch_poses_);
        ros::spinOnce();
        rate.sleep();
    }
}

int main(int argc,char** argv)
{
    ros::init(argc,argv,"mismatch_calculator");
    MismatchCalculator mismatch_calculator;
    mismatch_calculator.process();
    return 0;
}
