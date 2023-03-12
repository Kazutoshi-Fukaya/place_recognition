#ifndef DOM_PARAM_H_
#define DOM_PARAM_H_

#include <iostream>

namespace place_recognition
{
class DomParam
{
public:
	DomParam() :
		name(std::string("")), dom(0.0), object_size(0), appearance_count(0), disappearance_count(0), observations_count(0) {}
	
	DomParam(std::string _name,double _dom,int _object_size,int _appearance_count,int _disappearance_count,int _observations_count) :
		name(_name), dom(_dom), object_size(_object_size), appearance_count(_appearance_count), disappearance_count(_disappearance_count), observations_count(_observations_count) {}

	std::string name;
	double dom;
	int object_size;	// Number of objects on the map
	int appearance_count;	
	int disappearance_count;
	int observations_count;

private:
};
} // namespace place_recognition

#endif	// DOM_PARAMS_H_