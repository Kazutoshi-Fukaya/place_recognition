#ifndef VOCABULARY_H_
#define VOCABULARY_H_

#include <iostream>
#include <ostream>
#include <algorithm>
#include <numeric>

#include <opencv2/core/core.hpp>

#include "bow/bow_vector/bow_vector.h"
#include "bow/scoring_object/scoring_object.h"
#include "bow/vocabulary/node.h"
#include "bow/descriptors_manipulator/descriptors_manipulator.h"

namespace place_recognition
{
class Vocabulary
{
public:
    Vocabulary();
    Vocabulary(int k,int L,WeightingType weighting_type,ScoringType scoring_type);

    virtual void create(std::vector<cv::Mat>& training_features);
    virtual void create(std::vector<std::vector<cv::Mat>>& training_features);
	// virtual void transform(std::vector<cv::Mat>& features,BowVector& v);
	virtual void transform(cv::Mat& feature,unsigned int& id);
	virtual void transform(cv::Mat& feature,unsigned int& id,double& weight);
	virtual void transform(cv::Mat& features,BowVector& v);

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
	
    // descriptors_manipulator
    DescriptorsManipulator* descriptors_manipulator_;

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