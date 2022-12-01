#include <ros/ros.h>
#include <sensor_msgs/Image.h>

class KeypointsDetector
{
public:
	KeypointsDetector();
	void process();

private:
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;

};

KeypointsDetector::KeypointsDetector() :
	private_nh_("~")
{

}

void KeypointsDetector::process() { ros::spin(); }

int main(int argc,char** argv)
{
	ros::init(argc,argv,"keypoints_detector");
	KeypointsDetector keypoints_detector;
	keypoints_detector.process();
	return 0;
}