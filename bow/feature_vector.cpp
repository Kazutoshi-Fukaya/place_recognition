#include "bow/feature_vector/feature_vector.h"

using namespace place_recognition;

FeatureVector::FeatureVector() {}

FeatureVector::~FeatureVector() {}

void FeatureVector::add_feature(unsigned int id,unsigned int i_feature)
{
	FeatureVector::iterator vit = this->lower_bound(id);
	if(vit != this->end() && vit->first == id) vit->second.emplace_back(i_feature);
	else{
		vit = this->insert(vit,FeatureVector::value_type(id,std::vector<unsigned int>()));
		vit->second.emplace_back(i_feature);
  	}
}