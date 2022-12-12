#ifndef FEATURE_VECTOR_H_
#define FEATURE_VECTOR_H_

#include "bow/bow_vector/bow_vector.h"

namespace place_recognition
{
class FeatureVector : public std::map<unsigned int,std::vector<unsigned int>>
{
public:
	FeatureVector();
	~FeatureVector();

	void add_feature(unsigned int id,unsigned int i_feature);

private:
};
}

#endif	// FEATURE_VECTOR_H_