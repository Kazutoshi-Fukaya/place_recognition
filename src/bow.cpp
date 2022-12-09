#include "bow/bow.h"

using namespace place_recognition;

int main(int argc,char** argv)
{
	ros::init(argc,argv,"bow");
	BoW bow;
	bow.process();
	return 0;
}