#ifndef VOCABULARY_H_
#define VOCABULARY_H_

#include <iostream>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <numeric>

#include <opencv2/core/core.hpp>

#include "bow/bow_vector/bow_vector.h"
#include "bow/scoring_object/scoring_object.h"
#include "bow/vocabulary/node.h"
#include "bow/descriptors_manipulator/descriptors_manipulator.h"
#include "bow/quicklz/quicklz.h"
#include "bow/feature_vector/feature_vector.h"

namespace place_recognition
{
class Vocabulary
{
public:
    Vocabulary();
    Vocabulary(int k,int L,WeightingType weighting_type,ScoringType scoring_type);
    Vocabulary(std::string& file_name);

    virtual void create(std::vector<cv::Mat>& training_features);
    virtual void create(std::vector<std::vector<cv::Mat>>& training_features);
	// virtual void transform(std::vector<cv::Mat>& features,BowVector& v);
	virtual void transform(cv::Mat& feature,unsigned int& id);
	virtual void transform(cv::Mat& feature,unsigned int& id,double& weight);
	virtual void transform(cv::Mat& features,BowVector& v);
    virtual void transform(std::vector<cv::Mat>& features,BowVector& v,FeatureVector& fv,int levelsup);
    virtual void transform(std::vector<cv::Mat>& features,BowVector& v);

	void save(std::string& file_name,bool binary_compressed = true);
	virtual void save(cv::FileStorage& fs,std::string& name);
	void to_stream(std::ostream& str,bool compressed = true);
    void from_stream(std::istream& str);
    void load_fromtxt(std::string& file_name);
    virtual void load(cv::FileStorage& fs,std::string& name);

    void load(std::string& file_name);
    bool load(std::istream& ist);


	double score(BowVector& a,BowVector& b);

	int get_branching_factor();
	int get_depth_levels();
	WeightingType get_weighting_type();
	ScoringType get_scoring_type();
	virtual unsigned int size();
	virtual bool empty();

	// for debug
	void get_info();

protected:
    void create_scoring_object();
    void get_features(std::vector<std::vector<cv::Mat>>& training_features,
                      std::vector<cv::Mat>& features);
    void HKmean_step(unsigned int parent_id,std::vector<cv::Mat>& descriptors,int current_level);
    virtual void initiate_clusters(std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters);
    void initiate_clusters_KMpp(std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters);
	void create_words();
	void set_node_weights(std::vector<std::vector<cv::Mat>>& training_features);
    
    virtual void transform(cv::Mat& feature,unsigned int& id,double& weight,unsigned int* nid,int levelsup = 0);

    // descriptors_manipulator
    DescriptorsManipulator* descriptors_manipulator_;

    // QuickLZ
    QuickLZ* quick_lz_;

    int m_k;                            // Branching factor
    int m_L;                            // Depth levels
    WeightingType m_weighting;          // Weighting method
    ScoringType m_scoring;              // Scoring method
    GeneralScoring* m_scoring_object;   // Object for computing scores
    std::vector<Node> m_nodes;          // Tree nodes
    std::vector<Node*> m_words;         // Words of the vocabulary (tree leaves)
};
}

#endif  // VOCABULARY_H_