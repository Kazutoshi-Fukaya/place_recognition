#include "bow/database/database.h"

using namespace place_recognition;

Database::Database(bool use_di,int di_levels) : 
	m_voc(NULL), m_use_di(use_di), m_dilevels(di_levels), m_nentries(0) {}
	
Database::Database(Vocabulary& voc,bool use_di,int di_levels) : 
	m_voc(NULL), m_use_di(use_di), m_dilevels(di_levels)
{
	set_vocabulary(voc);
	clear();
}

Database::Database(Database& database) : 
	m_voc(NULL) { *this = database; }

Database::Database(std::string& file_name) : 
	m_voc(NULL)
{
	load(file_name);
}

// Database::Database(const char *file_name) : 
	// m_voc(NULL)
// {
	// load(file_name);
// }

Database::~Database() { delete m_voc; }

Database& Database::operator=(Database& database)
{
	if(this != &database){
		m_dfile = database.m_dfile;
		m_dilevels = database.m_dilevels;
		m_ifile = database.m_ifile;
		m_nentries = database.m_nentries;
		m_use_di = database.m_use_di;
		if(database.m_voc != 0) set_vocabulary(*database.m_voc);
  	}
	return *this;
}

void Database::set_vocabulary(Vocabulary& voc)
{
	delete m_voc;
	m_voc = new Vocabulary(voc);
	clear();
}

void Database::set_vocabulary(Vocabulary& voc,bool use_di,int di_levels)
{
	m_use_di = use_di;
	m_dilevels = di_levels;
	delete m_voc;
	m_voc = new Vocabulary(voc);
	clear();
}

Vocabulary* Database::get_vocabulary() { return m_voc; }

void Database::allocate(int nd,int ni)
{
	// m_ifile already contains |words| items
	if(ni > 0){
		for(auto rit = m_ifile.begin(); rit != m_ifile.end(); rit++){
			int n = (int)rit->size();
			if(ni > n){
				rit->resize(ni);
				rit->resize(n);
      		}
    	}
  	}
	
	if(m_use_di && (int)m_dfile.size() < nd) m_dfile.resize(nd); 
}

unsigned int Database::add(std::vector<cv::Mat>& features,BowVector* bowvec,FeatureVector* fvec)
{
	BowVector aux;
	BowVector& v = (bowvec ? *bowvec : aux);
	
	if(m_use_di && fvec != NULL){
		m_voc->transform(features,v,*fvec,m_dilevels);	// with features
		return add(v,*fvec);
  	}
  	else if(m_use_di){
		FeatureVector fv;
    	m_voc->transform(features,v,fv,m_dilevels);		// with features
    	return add(v, fv);
  	}
	else if(fvec != NULL){
		m_voc->transform(features,v,*fvec,m_dilevels);	// with features
    	return add(v);
  	}
	else{
		m_voc->transform(features, v);	// with features
		return add(v);
  	}
}

unsigned int Database::add(cv::Mat& features,BowVector* bowvec,FeatureVector* fvec)
{
	std::vector<cv::Mat> vf(features.rows);
    for(int r = 0; r < features.rows; r++) vf[r] = features.rowRange(r,r+1);
    return add(vf,bowvec,fvec);
}

unsigned int Database::add(BowVector& v,const FeatureVector &fv)
{
	unsigned int entry_id = m_nentries++;
	
	BowVector::const_iterator vit;
	std::vector<unsigned int>::const_iterator iit;
	if(m_use_di){
		// update direct file
		if(entry_id == m_dfile.size()){
			m_dfile.emplace_back(fv);
    	}
    	else{
			m_dfile[entry_id] = fv;
    	}
  	}
	
	// update inverted file
  	for(vit = v.begin(); vit != v.end(); vit++){
		const unsigned int& word_id = vit->first;
		const double& word_weight = vit->second;
		
		std::list<IFPair>& ifrow = m_ifile[word_id];
		ifrow.emplace_back(IFPair(entry_id,word_weight));
  	}
	
	return entry_id;
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

void Database::query(std::vector<cv::Mat>& features,QueryResults& ret,int max_results,int max_id)
{
	BowVector vec;
	m_voc->transform(features,vec);
	query(vec,ret,max_results,max_id);
}

void Database::query(cv::Mat& features,QueryResults& ret,int max_results,int max_id)
{
	std::vector<cv::Mat> vf(features.rows);
	for(int r = 0; r < features.rows; r++) vf[r] = features.rowRange(r,r+1);
    query(vf,ret,max_results,max_id);
}

void Database::query(BowVector& vec,QueryResults& ret,int max_results,int max_id)
{
	ret.resize(0);
	switch(m_voc->get_scoring_type())
	{
		case L1_NORM:
			query_L1(vec,ret,max_results,max_id);
      		break;
		
		case L2_NORM:
			query_L2(vec,ret,max_results,max_id);
      		break;

    	case CHI_SQUARE:
      		query_ChiSquare(vec,ret,max_results,max_id);
      		break;

    	case KL:
      		query_KL(vec,ret,max_results,max_id);
      		break;

    	case BHATTACHARYYA:
      		query_Bhattacharyya(vec,ret,max_results,max_id);
      		break;

    	case DOT_PRODUCT:
      		query_DotProduct(vec,ret,max_results,max_id);
      	break;
  	}
}

FeatureVector& Database::retrieve_features(unsigned int id)
{
	assert(id < size());
	return m_dfile[id];
}

void Database::save(std::string& file_name)
{
	cv::FileStorage fs(file_name.c_str(), cv::FileStorage::WRITE);
	if(!fs.isOpened()) throw std::string("Could not open file ") + file_name;
	save(fs);
}

void Database::save(cv::FileStorage& fs,const std::string &name)
{
	m_voc->save(fs);
	fs << name << "{";
	fs << "nEntries" << m_nentries;
	fs << "usingDI" << (m_use_di ? 1 : 0);
	fs << "diLevels" << m_dilevels;
	
	fs << "invertedIndex" << "[";
	for(auto iit = m_ifile.begin(); iit != m_ifile.end(); iit++){
		// word of IF
		fs << "[";
    	for(auto irit = iit->begin(); irit != iit->end(); irit++){
			fs << "{:"
        	   << "imageId" << (int)irit->entry_id
               << "weight" << irit->word_weight
               << "}";
    	}
    	fs << "]";
  	}
	fs << "]";
	
	fs << "directIndex" << "[";
	for(auto dit = m_dfile.begin(); dit != m_dfile.end(); dit++){
		// entry of DF
		fs << "[";
		for(auto drit = dit->begin(); drit != dit->end(); drit++){
			unsigned int nid = drit->first;
      		const std::vector<unsigned int>& features = drit->second;
			
			// save info of last_nid
      		fs << "{";
      		fs << "nodeId" << (int)nid;
      		fs << "features" << "["
               << *(const std::vector<int>*)(&features) << "]";
      		fs << "}";
    	}
		fs << "]";
  	}
	fs << "]";
	fs << "}";
}

void Database::load(std::string& file_name)
{
	cv::FileStorage fs(file_name.c_str(), cv::FileStorage::READ);
	if(!fs.isOpened()) throw std::string("Could not open file ") + file_name;
	load(fs);
}

void Database::load(cv::FileStorage& fs,const std::string& name)
{
	if(!m_voc) m_voc = new Vocabulary;
	m_voc->load(fs);
	
	// load database now
	clear();	// resizes inverted file
	
	cv::FileNode fdb = fs[name];
	m_nentries = (int)fdb["nEntries"];
  	m_use_di = (int)fdb["usingDI"] != 0;
  	m_dilevels = (int)fdb["diLevels"];

  	cv::FileNode fn = fdb["invertedIndex"];
  	for(unsigned int wid = 0; wid < fn.size(); wid++){
    	cv::FileNode fw = fn[wid];
		for(unsigned int i = 0; i < fw.size(); i++){
			unsigned int eid = (int)fw[i]["imageId"];
      		double v = fw[i]["weight"];

      		m_ifile[wid].push_back(IFPair(eid, v));
    	}
  	}
	
	if(m_use_di){
		fn = fdb["directIndex"];

    	m_dfile.resize(fn.size());
    	assert(m_nentries == (int)fn.size());

    	FeatureVector::iterator dit;
    	for(unsigned int eid = 0; eid < fn.size(); eid++){
			cv::FileNode fe = fn[eid];
			
			m_dfile[eid].clear();
      		for(unsigned int i = 0; i < fe.size(); i++){
				unsigned int nid = (int)fe[i]["nodeId"];
				dit = m_dfile[eid].insert(m_dfile[eid].end(),std::make_pair(nid,std::vector<unsigned int>()));
				
				cv::FileNode ff = fe[i]["features"][0];
        		dit->second.reserve(ff.size());

        		cv::FileNodeIterator ffit;
        		for(ffit = ff.begin(); ffit != ff.end(); ffit++){
          			dit->second.emplace_back((int)*ffit);
        		}
      		}
    	}
  	}
}

void Database::query_L1(BowVector& vec,QueryResults& ret,int max_results,int max_id)
{
	BowVector::const_iterator vit;
	
	std::map<unsigned int,double> pairs;
	std::map<unsigned int,double>::iterator pit;
	for(vit = vec.begin(); vit != vec.end(); vit++){
		const unsigned int word_id = vit->first;
		const double& qvalue = vit->second;
		
		const std::list<IFPair>& row = m_ifile[word_id];
		for(auto rit = row.begin(); rit != row.end(); rit++){
			const unsigned int entry_id = rit->entry_id;
			const double& dvalue = rit->word_weight;
			
			if((int)entry_id < max_id || max_id == -1){
				double value = std::fabs(qvalue - dvalue) - std::fabs(qvalue) - std::fabs(dvalue);
				
				pit = pairs.lower_bound(entry_id);
				if(pit != pairs.end() && !(pairs.key_comp()(entry_id,pit->first))){
					pit->second += value;
        		}
        		else{
					pairs.insert(pit,std::map<unsigned int,double>::value_type(entry_id,value));
        		}
      		}
		}
	}
	
	// move to vector
  	ret.reserve(pairs.size());
  	for(pit = pairs.begin(); pit != pairs.end(); pit++){
		ret.emplace_back(Result(pit->first,pit->second));
  	}
	std::sort(ret.begin(),ret.end());
	
	// cut vector
	if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);
	
	QueryResults::iterator qit;
	for(qit = ret.begin(); qit != ret.end(); qit++) qit->score = -qit->score/2.0;
}

void Database::query_L2(BowVector& vec,QueryResults& ret,int max_results,int max_id)
{
	BowVector::const_iterator vit;
	
	std::map<unsigned int,double> pairs;
	std::map<unsigned int,double>::iterator pit;
	for(vit = vec.begin(); vit != vec.end(); vit++){
		const unsigned int word_id = vit->first;
		const double& qvalue = vit->second;
		
		const std::list<IFPair>& row = m_ifile[word_id];
		for(auto rit = row.begin(); rit != row.end(); rit++){
			const unsigned int entry_id = rit->entry_id;
			const double& dvalue = rit->word_weight;
			
			if((int)entry_id < max_id || max_id == -1){
				double value = - qvalue*dvalue;	// minus sign for sorting trick
				
				pit = pairs.lower_bound(entry_id);
				if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
					pit->second += value;
				}
        		else{
					pairs.insert(pit,std::map<unsigned int,double>::value_type(entry_id,value));
				}
			}
		}
	}
	
	// move to vector
  	ret.reserve(pairs.size());
  	for(pit = pairs.begin(); pit != pairs.end(); pit++){
    	ret.emplace_back(Result(pit->first,pit->second));
  	}
	std::sort(ret.begin(),ret.end());
	
	// cut vector
	if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);
	
	QueryResults::iterator qit;
	for(qit = ret.begin(); qit != ret.end(); qit++){
		if(qit->score <= -1.0){
			// rounding error
			qit->score = 1.0;
		}
		else{
			// [0..1]
			qit->score = 1.0 - std::sqrt(1.0 + qit->score);
		}
  }
}

void Database::query_ChiSquare(BowVector& vec,QueryResults& ret,int max_results,int max_id)
{
	BowVector::const_iterator vit;
	
	std::map<unsigned int,std::pair<double,int>> pairs;
	std::map<unsigned int,std::pair<double,int>>::iterator pit;

	std::map<unsigned int,std::pair<double,double>> sums;
  	std::map<unsigned int,std::pair<double,double>>::iterator sit;
	
	for(vit = vec.begin(); vit != vec.end(); vit++){
    	const unsigned int word_id = vit->first;
    	const double& qvalue = vit->second;

    	const std::list<IFPair>& row = m_ifile[word_id];
		
		for(auto rit = row.begin(); rit != row.end(); rit++){
			const unsigned int entry_id = rit->entry_id;
			const double& dvalue = rit->word_weight;
			
			if((int)entry_id < max_id || max_id == -1){
				double value = 0;
				// words may have weight zero
				if(qvalue + dvalue != 0.0) value = - qvalue*dvalue/(qvalue + dvalue);
				
				pit = pairs.lower_bound(entry_id);
				sit = sums.lower_bound(entry_id);
				if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
					pit->second.first += value;
          			pit->second.second += 1;
          			sit->second.first += qvalue;
          			sit->second.second += dvalue;
        		}
				else{
					pairs.insert(pit,std::map<unsigned int,std::pair<double,int>>::value_type(entry_id,std::make_pair(value,1)));
					sums.insert(sit,std::map<unsigned int,std::pair<double,double>>::value_type(entry_id,std::make_pair(qvalue,dvalue)));
        		}
      		}
    	}
  	}
	
	// move to vector
  	ret.reserve(pairs.size());
  	sit = sums.begin();
  	for(pit = pairs.begin(); pit != pairs.end(); pit++, sit++){
		if(pit->second.second >= MIN_COMMON_WORDS){
			ret.emplace_back(Result(pit->first,pit->second.first));
			ret.back().nWords = pit->second.second;
      		ret.back().sumCommonVi = sit->second.first;
      		ret.back().sumCommonWi = sit->second.second;
      		ret.back().expectedChiScore = 2*sit->second.second/(1 + sit->second.second);
    	}
  	}
	std::sort(ret.begin(), ret.end());
	
	// cut vector
	if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);
	
	// complete and scale score to [0 worst .. 1 best]
	QueryResults::iterator qit;
	for(qit = ret.begin(); qit != ret.end(); qit++){
		// this takes the 4 into account
    	qit->score = - 2.*qit->score;	// [0..1]
		qit->chiScore = qit->score;
  	}
}

void Database::query_KL(BowVector& vec,QueryResults& ret,int max_results,int max_id)
{
	BowVector::const_iterator vit;
	
	std::map<unsigned int,double> pairs;
	std::map<unsigned int,double>::iterator pit;
	for(vit = vec.begin(); vit != vec.end(); vit++){
		const unsigned int word_id = vit->first;
		const double& vi = vit->second;

    	const std::list<IFPair>& row = m_ifile[word_id];
		for(auto rit = row.begin(); rit != row.end(); ++rit){
			const unsigned int entry_id = rit->entry_id;
			const double& wi = rit->word_weight;
			
			if((int)entry_id < max_id || max_id == -1){
				double value = 0;
				if(vi != 0 && wi != 0) value = vi*std::log(vi/wi);
				
				pit = pairs.lower_bound(entry_id);
				if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
					pit->second += value;
        		}
        		else{
					pairs.insert(pit,std::map<unsigned int,double>::value_type(entry_id, value));
        		}
      		}
		}
	}
	
	// complete scores and move to vector
  	ret.reserve(pairs.size());
  	for(pit = pairs.begin(); pit != pairs.end(); ++pit){
    	unsigned int eid = pit->first;
    	double value = 0.0;

    	for(vit = vec.begin(); vit != vec.end(); ++vit){
			const double& vi = vit->second;
			const std::list<IFPair>& row = m_ifile[vit->first];

      		if(vi != 0){
				if(row.end() == std::find(row.begin(),row.end(),eid)){
					value += vi*(std::log(vi) - LOG_EPS_);
        		}
      		}
    	}
		pit->second += value;
		
		// to vector
		ret.emplace_back(Result(pit->first, pit->second));
	}
	std::sort(ret.begin(), ret.end());
	
	// cut vector
  	if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);
}

void Database::query_Bhattacharyya(BowVector& vec,QueryResults& ret,int max_results,int max_id)
{
	BowVector::const_iterator vit;
	
	std::map<unsigned int,std::pair<double,int> > pairs;
	std::map<unsigned int,std::pair<double,int> >::iterator pit;
	
	for(vit = vec.begin(); vit != vec.end(); vit++){
		const unsigned int word_id = vit->first;
    	const double& qvalue = vit->second;

    	const std::list<IFPair>& row = m_ifile[word_id];
		for(auto rit = row.begin(); rit != row.end(); rit++){
			const unsigned int entry_id = rit->entry_id;
			const double& dvalue = rit->word_weight;
			
			if((int)entry_id < max_id || max_id == -1){
				double value = std::sqrt(qvalue*dvalue);

        		pit = pairs.lower_bound(entry_id);
        		if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
          			pit->second.first += value;
          			pit->second.second += 1;
        		}
        		else{
          			pairs.insert(pit,std::map<unsigned int,std::pair<double,int> >::value_type(entry_id,std::make_pair(value,1)));
        		}
      		}
    	}
  	}
	
	// move to vector
  	ret.reserve(pairs.size());
  	for(pit = pairs.begin(); pit != pairs.end(); pit++){
		if(pit->second.second >= MIN_COMMON_WORDS){
			ret.emplace_back(Result(pit->first, pit->second.first));
      		ret.back().nWords = pit->second.second;
      		ret.back().bhatScore = pit->second.first;
    	}
  	}
	std::sort(ret.begin(), ret.end(), Result::gt);
	
	// cut vector
  	if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);
}

void Database::query_DotProduct(BowVector& vec,QueryResults& ret,int max_results,int max_id)
{
	BowVector::const_iterator vit;
	
	std::map<unsigned int,double> pairs;
	std::map<unsigned int,double>::iterator pit;
	for(vit = vec.begin(); vit != vec.end(); vit++){
    	const unsigned int word_id = vit->first;
    	const double& qvalue = vit->second;

    	const std::list<IFPair>& row = m_ifile[word_id];
		for(auto rit = row.begin(); rit != row.end(); ++rit){
      		const unsigned int entry_id = rit->entry_id;
      		const double& dvalue = rit->word_weight;

      		if((int)entry_id < max_id || max_id == -1){
				double value;
				if(this->m_voc->get_weighting_type() == BINARY) value = 1;
        		else value = qvalue * dvalue;
				
				pit = pairs.lower_bound(entry_id);
				if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first))){
					pit->second += value;
        		}
        		else{
					pairs.insert(pit,std::map<unsigned int,double>::value_type(entry_id,value));
        		}
      		}
    	}
  	}
	
	// move to vector
  	ret.reserve(pairs.size());
  	for(pit = pairs.begin(); pit != pairs.end(); pit++){
		ret.emplace_back(Result(pit->first,pit->second));
  	}
	std::sort(ret.begin(), ret.end(), Result::gt);
	
	// cut vector
  	if(max_results > 0 && (int)ret.size() > max_results) ret.resize(max_results);
}

// for debug
void Database::get_vocabulary_info() { m_voc->get_info(); }

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