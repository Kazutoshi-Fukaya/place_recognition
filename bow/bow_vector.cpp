#include "bow/bow_vector/bow_vector.h"

using namespace place_recognition;

BowVector::BowVector() {}

BowVector::~BowVector() {}

void BowVector::add_weight(unsigned int id,double value)
{
	BowVector::iterator vit = this->lower_bound(id);
	if(vit != this->end() && !(this->key_comp())(id,vit->first)){
		vit->second += value;
	}
	else{
		this->insert(vit,BowVector::value_type(id,value));
	}
}

void BowVector::add_if_not_exist(unsigned int id,double value)
{
	BowVector::iterator vit = this->lower_bound(id);
	if(vit == this->end() || (this->key_comp())(id,vit->first)){
		this->insert(vit,BowVector::value_type(id,value));
	}
}

void BowVector::normalize(LNorm norm_type)
{
	double norm = 0.0;
	BowVector::iterator it;

	if(norm_type == L1){
		for(it = this->begin(); it != this->end(); it++){
			norm += std::fabs(it->second);
		}
	}
	else{
		for(it = this->begin(); it != this->end(); it++){
			norm += it->second*it->second;
			norm = std::sqrt(norm);
		}
	}

	if(norm > 0.0){
		for(it = this->begin(); it != this->end(); it++){
			it->second /= norm;
		}
	}
}

uint16_t BowVector::get_signature()
{
	uint16_t signature = 0;
	for(auto bow : *this) signature += bow.first + 1e6*bow.second;
	return signature;
}