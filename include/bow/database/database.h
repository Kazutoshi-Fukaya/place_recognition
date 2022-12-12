#ifndef DATABASE_H_
#define DATABASE_H_

#include <list>

#include "bow/vocabulary/vocabulary.h"
#include "bow/database/if_pair.h"
#include "bow/feature_vector/feature_vector.h"
#include "bow/query_results/query_results.h"

namespace place_recognition
{
class Database
{
public:
	Database(bool use_di = true,int di_levels = 0);
	Database(Vocabulary& voc,bool use_di = true,int di_levels = 0);
	Database(Database& database);
	Database(std::string& file_name);
	// Database(const char *file_name);
	
	virtual ~Database();
	Database& operator=(Database& database);
	
	void set_vocabulary(Vocabulary& voc);
	void set_vocabulary(Vocabulary& voc,bool use_di,int di_levels = 0);
	Vocabulary* get_vocabulary();
	void allocate(int nd = 0,int ni = 0);
	
	// add
	unsigned int add(std::vector<cv::Mat>& features,BowVector* bowvec = NULL,FeatureVector* fvec = NULL);
	unsigned int add(cv::Mat& features,BowVector* bowvec = NULL,FeatureVector* fvec = NULL);
	unsigned int add(BowVector& vec,const FeatureVector& fec = FeatureVector());

	// utils
	void clear();
	unsigned int size();
	bool using_direct_index();
	int get_direct_index_levels();
	
	void query(std::vector<cv::Mat>& features,QueryResults& ret,int max_results = 1,int max_id = -1);
	void query(cv::Mat &features,QueryResults& ret,int max_results = 1,int max_id = -1);
	void query(BowVector& vec,QueryResults& ret,int max_results = 1,int max_id = -1);
	
	FeatureVector& retrieve_features(unsigned int id);
	
	void save(std::string& file_name);
	virtual void save(cv::FileStorage& fs,const std::string& name = "database");
	void load(std::string& file_name);
	virtual void load(cv::FileStorage& fs,const std::string& name = "database");

	// for debug
	void get_vocabulary_info();
	void get_info();

protected:
	// Query with L1 scoring
	void query_L1(BowVector& vec,QueryResults& ret,int max_results,int max_id);
	
	// Query with L2 scoring
	void query_L2(BowVector& vec,QueryResults& ret,int max_results,int max_id);
	
	// Query with Chi square scoring
  	void query_ChiSquare(BowVector& vec,QueryResults& ret,int max_results,int max_id);
	
	// Query with Bhattacharyya scoring
  	void query_Bhattacharyya(BowVector& vec,QueryResults& ret,int max_results,int max_id);
	
	// Query with KL divergence scoring  
	void query_KL(BowVector& vec,QueryResults& ret,int max_results,int max_id);
	
	// Query with dot product scoring
	void query_DotProduct(BowVector& vec,QueryResults& ret,int max_results,int max_id);

protected:
	// For query functions
	const int MIN_COMMON_WORDS = 5;
	
	const double LOG_EPS_ = std::log(DBL_EPSILON);

	// buffer
	Vocabulary* m_voc;	                    // Associated vocabulary
	bool m_use_di;	                        // Flag to use direct index
	int m_dilevels;                         // Levels to go up the vocabulary tree to select nodes to store in the direct index
	std::vector<std::list<IFPair>> m_ifile; // Inverted file (must have size() == |words|)
	std::vector<FeatureVector> m_dfile;     // Direct file (resized for allocation)
	int m_nentries;                         // Number of valid entries in m_dfile
};
}

#endif	// DATABASE_H_