#ifndef DATABASE_H_
#define DATABASE_H_

#include <list>

#include "bow/vocabulary/vocabulary.h"
#include "bow/database/if_pair.h"
#include "bow/feature_vector/feature_vector.h"

namespace place_recognition
{
class Database
{
public:
	Database(Vocabulary& voc,bool use_di,int di_levels);
	void set_vocabulary(Vocabulary& voc);
	void clear();
	unsigned int size();
	bool using_direct_index();
	int get_direct_index_levels();
	void get_vocabulary_info();

	unsigned int add(cv::Mat& features,BowVector* bowvec = NULL,FeatureVector* fvec = NULL);
	unsigned int add(std::vector<cv::Mat>& features,BowVector* bowvec = NULL,FeatureVector* fvec = NULL);
	unsigned int add(BowVector& vec,const FeatureVector& fec = FeatureVector());
	
	// for debug
	void get_info();

private:

protected:
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