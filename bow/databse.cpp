#include "bow/database/database.h"

using namespace place_recognition;

Database::Database(Vocabulary& voc,bool use_di,int di_levels) : 
	m_voc(NULL), m_use_di(use_di), m_dilevels(di_levels)
{
	set_vocabulary(voc);
	clear();
}

void Database::set_vocabulary(Vocabulary& voc)
{
	delete m_voc;
	m_voc = new Vocabulary(voc);
	clear();
}

void Database::clear()
{
	// resize vectors
	m_ifile.resize(0);
	m_ifile.resize(m_voc->size());
	m_dfile.resize(0);
	m_nentries = 0;
}

unsigned int Database::size() { return m_nentries; }

bool Database::using_direct_index() { return m_use_di; }

int Database::get_direct_index_levels() { return m_dilevels; }

void Database::get_vocabulary_info() { m_voc->get_info(); }

unsigned int Database::add(cv::Mat& features,BowVector* bowvec,FeatureVector* fvec)
{
	std::vector<cv::Mat> vf(features.rows);
    for(int r = 0; r < features.rows; r++) vf[r] = features.rowRange(r,r + 1);
    return add(vf,bowvec,fvec);
}

unsigned int Database::add(std::vector<cv::Mat>& features,BowVector* bowvec,FeatureVector* fvec)
{
	BowVector aux;
	BowVector& v = (bowvec ? *bowvec : aux);
	
	if(m_use_di && fvec != NULL){
		m_voc->transform(features,v,*fvec,m_dilevels);
		return add(v,*fvec);
  	}	
  	else if(m_use_di){
		FeatureVector fv;
		m_voc->transform(features,v,fv,m_dilevels);
    	return add(v,fv);
  	}
  	else if(fvec != NULL){
		m_voc->transform(features,v,*fvec,m_dilevels);
    	return add(v);
  	}
  	else{
		m_voc->transform(features,v);
    	return add(v);
  	}
}

unsigned int Database::add(BowVector& v,const FeatureVector& fv)
{
	unsigned int entry_id = m_nentries++;
	BowVector::const_iterator vit;
	std::vector<unsigned int>::const_iterator iit;
	
	if(m_use_di){
		// update direct file
		if(entry_id == m_dfile.size()) m_dfile.emplace_back(fv);
		else m_dfile[entry_id] = fv;
  	}
	
	// update inverted file
  	for(vit = v.begin(); vit != v.end(); vit++){
		const unsigned int& word_id = vit->first;
		const double& word_weight = vit->second;
		
		std::list<IFPair>& ifrow = m_ifile[word_id];
		ifrow.push_back(IFPair(entry_id,word_weight));
  	}
	return entry_id;
}

// for debug
void Database::get_info()
{
	std::cout << "Entries = " << this->size() << std::endl;
	std::cout << "Using direct index = ";
	
	if(using_direct_index()){
		std::cout << "yes" << std::endl;
		std::cout << "Direct index levels = " << get_direct_index_levels() << std::endl;
	}
	else std::cout << "no" << std::endl;

	std::cout << "Vocabulary: " << std::endl;
	get_vocabulary_info();
}