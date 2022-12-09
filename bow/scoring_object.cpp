#include "bow/scoring_object/scoring_object.h"

using namespace place_recognition;

GeneralScoring::GeneralScoring() {}

GeneralScoring::GeneralScoring(std::string name)
{
	if(name == "L1Scoring"){
		set_params(name,true,L1);
	}
	else if(name == "L2Scoring"){
		set_params(name,true,L2);
	}
	else if(name == "ChiSquareScoring"){
		set_params(name,true,L1);
	}
	else if(name == "KLScoring"){
		set_params(name,true,L1);
	}
	else if(name == "BhattacharyyaScoring"){
		set_params(name,true,L1);
	}
	else if(name == "DotProductScoring"){
		set_params(name,false,L1);
	}
	else{
		std::cerr << "No applicable mode." << std::endl;
		std::cerr << "Please select 'L1Scoring', 'L2Scoring', 'ChiSquareScoring', 'KLScoring', 'BhattacharyyaScoring' or 'DotProductScoring'" << std::endl;
		std::cerr << "Set 'L1Scoring'" << std::endl;
		name = "L1Scoring";
		set_params(name,true,L1);
	}
}

GeneralScoring::~GeneralScoring() {}

void GeneralScoring::set_params(std::string name,bool must_normalize,LNorm l_norm)
{
	NAME = name;
	MUST_NORMALIZE = must_normalize;
	L_NORM = l_norm;
}

double GeneralScoring::score(BowVector& v,BowVector& w)
{
	if(NAME == "L1Scoring"){
		l1_score(v,w);
	}
	else if(NAME == "L2Scoring"){
		l2_score(v,w);
	}
	else if(NAME == "ChiSquareScoring"){
		chi_square_score(v,w);
	}
	else if(NAME == "KLScoring"){
		kl_score(v,w);
	}
	else if(NAME == "BhattacharyyaScoring"){
		bhattacharyya_score(v,w);
	}
	else if(NAME == "DotProductScoring"){
		dot_product_score(v,w);
	}
	else{
		std::cerr << "bug occurrence!" << std::endl;
	}
}

bool GeneralScoring::must_normalize(LNorm& l_norm)
{
	l_norm = L_NORM; 
	return MUST_NORMALIZE; 
}

double GeneralScoring::l1_score(BowVector& v,BowVector& w)
{
	BowVector::const_iterator v_it, w_it;
  	const BowVector::const_iterator v_end = v.end();
  	const BowVector::const_iterator w_end = w.end();
	v_it = v.begin();
	w_it = w.begin();

	double score = 0;
	while(v_it != v_end && w_it != w_end){
		const double& vi = v_it->second;
		const double& wi = w_it->second;
		if(v_it->first == w_it->first){
			score += std::fabs(vi - wi) - std::fabs(vi) - std::fabs(wi);
			v_it++;
			w_it++;
		}
		else if(v_it->first < w_it->first){
			// move v forward
			v_it = v.lower_bound(w_it->first);
		}
		else
		{
			// move w forward
			w_it = w.lower_bound(v_it->first);
		}
	}
	
	score = -score/2.0;
	return score; // [0..1]
}

double GeneralScoring::l2_score(BowVector& v,BowVector& w)
{
	BowVector::const_iterator v_it, w_it;
	const BowVector::const_iterator v_end = v.end();
	const BowVector::const_iterator w_end = w.end();
	v_it = v.begin();
	w_it = w.begin();
  
  	double score = 0;
	while(v_it != v_end && w_it != w_end){
		const double& vi = v_it->second;
		const double& wi = w_it->second;
		
		if(v_it->first == w_it->first){
			score += vi*wi;
			v_it++;
			w_it++;
    	}
    	else if(v_it->first < w_it->first){
			// move v1 forward
			v_it = v.lower_bound(w_it->first);
		}
		else{
			// move v2 forward
			w_it = w.lower_bound(v_it->first);
		}
	}
	if(score >= 1){
		// rounding errors
	  	score = 1.0;
	}
	else score = 1.0 - std::sqrt(1.0 - score); // [0..1]

  	return score;
}

double GeneralScoring::chi_square_score(BowVector& v,BowVector& w)
{
	BowVector::const_iterator v_it, w_it;
	const BowVector::const_iterator v_end = v.end();
	const BowVector::const_iterator w_end = w.end();
	v_it = v.begin();
	w_it = w.begin();
	
  	// all the items are taken into account
	double score = 0;
  	while(v_it != v_end && w_it != w_end){
		const double& vi = v_it->second;
		const double& wi = w_it->second;
		if(v_it->first == w_it->first){
			// we move the -4 out
      		if(vi + wi != 0.0) score += vi*wi/(vi + wi);
			v_it++;
			w_it++;
    	}
    	else if(v_it->first < w_it->first){
			v_it = v.lower_bound(w_it->first);
    	}
    	else{
			w_it = w.lower_bound(v_it->first);
    	}
  	}
    
  	// this takes the -4 into account
  	score = 2. *score; // [0..1]
	return score;
}

double GeneralScoring::kl_score(BowVector& v,BowVector& w)
{
	BowVector::const_iterator v_it, w_it;
	const BowVector::const_iterator v_end = v.end();
	const BowVector::const_iterator w_end = w.end();
	v_it = v.begin();
	w_it = w.begin();
	
	// all the items or v are taken into account
  	double score = 0;
  	while(v_it != v_end && w_it != w_end){
		const double& vi = v_it->second;
    	const double& wi = w_it->second;
		if(v_it->first == w_it->first){
      		if(vi != 0 && wi != 0) score += vi*std::log(vi/wi);
			v_it++;
      		w_it++;
   	 	}
    	else if(v_it->first < w_it->first){
			score += vi*(std::log(vi) - LOG_EPS_);
			v_it++;
    	}
    	else{
			// move v2_it forward, do not add any score
    		w_it = w.lower_bound(v_it->first);
    	}
  	}
  
  	// sum rest of items of v
  	for(; v_it != v_end; v_it++){
		if(v_it->second != 0){
			score += v_it->second*(std::log(v_it->second) - LOG_EPS_);
		}
	}
  	
	return score; // cannot be scaled
}

double GeneralScoring::bhattacharyya_score(BowVector& v,BowVector& w)
{
	BowVector::const_iterator v_it, w_it;
	const BowVector::const_iterator v_end = v.end();
	const BowVector::const_iterator w_end = w.end();
	v_it = v.begin();
	w_it = v.begin();
	
	double score = 0;
	while(v_it != v_end && w_it != w_end){
		const double& vi = v_it->second;
		const double& wi = w_it->second;
		if(v_it->first == w_it->first){
			score += std::sqrt(vi*wi);
			v_it++;
			w_it++;
    	}
    	else if(v_it->first < w_it->first){
			// move v1 forward
			v_it = v.lower_bound(w_it->first);
		}else{
			// move v2 forward
			w_it = w.lower_bound(w_it->first);
		}
  	}
	
	return score; // already scaled
}

double GeneralScoring::dot_product_score(BowVector& v,BowVector& w)
{
	BowVector::const_iterator v_it, w_it;
	const BowVector::const_iterator v_end = v.end();
	const BowVector::const_iterator w_end = w.end();
	v_it = v.begin();
	w_it = w.begin();
	
	double score = 0;
	while(v_it != v_end && w_it != w_end){
		const double& vi = v_it->second;
		const double& wi = w_it->second;
		
		if(v_it->first == w_it->first){
			score += vi*wi;
			v_it++;
			w_it++;
   	 	}
    	else if(v_it->first < w_it->first){
			// move v1 forward
      		v_it = v.lower_bound(w_it->first);
		}
		else{
      		// move v2 forward
			w_it = w.lower_bound(v_it->first);
    	}
  	}
	
	return score; // cannot scale
}