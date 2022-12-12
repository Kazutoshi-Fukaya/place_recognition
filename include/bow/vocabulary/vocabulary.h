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
    // Vocabulary();
    Vocabulary(int k = 10,int L = 5,WeightingType weighting = TF_IDF,ScoringType scoring = L1_NORM);
    Vocabulary(std::string& file_name);
    Vocabulary(char* file_name);
    Vocabulary(std::istream& stream);
    Vocabulary(Vocabulary& voc);
    
    Vocabulary& operator=(Vocabulary& voc);
    virtual ~Vocabulary();
    
    // create
    virtual void create(std::vector<cv::Mat>& training_features);
    virtual void create(std::vector<std::vector<cv::Mat>>& training_features);
    virtual void create(std::vector<std::vector<cv::Mat>>& training_features,int k,int L);
    virtual void create(std::vector<std::vector<cv::Mat>>& training_features,int k,int L,WeightingType weighting,ScoringType scoring);
    
    // utils
    virtual unsigned int size();
    virtual bool empty();
    void clear();
    
    // Transforms a set of descriptores
    virtual void transform(std::vector<cv::Mat>& features,BowVector& v);
    virtual void transform(cv::Mat& features,BowVector& v);
    virtual void transform(std::vector<cv::Mat>& features,BowVector& v,FeatureVector& fv,int levelsup);
    virtual unsigned int transform(cv::Mat& feature);
    
    // score
    double score(BowVector& a,BowVector &b);
    
    // get utils
    virtual unsigned int get_parent_node(unsigned int wid,int levelsup);
    void get_words_from_node(unsigned int nid,std::vector<unsigned int>& words);
    int get_branching_factor();
    int get_depth_levels();
    float get_effective_levels();
    virtual cv::Mat get_word(unsigned int wid);
    virtual double get_word_weight(unsigned int wid);
    WeightingType get_weighting_type();
    ScoringType get_scoring_type();
    
    // set utils
    void set_weighting_type(WeightingType type);
    void set_scoring_type(ScoringType type);
    
    void save(std::string& file_name,bool binary_compressed = true);
    virtual void save(cv::FileStorage& fs,const std::string& name = "vocabulary");
    void load(std::string& file_name);
    bool load(std::istream& stream);
    virtual void load(cv::FileStorage& fs,const std::string& name = "vocabulary");
    
    virtual int stop_words(double min_weight);
    int get_descritor_size();
    int get_descritor_type();
    
    // stream
    void to_stream(std::ostream& str,bool compressed = true);
    void from_stream(std::istream& str);

	// for debug
	void get_info();

protected:
    void create_scoring_object();
    void get_features(std::vector<std::vector<cv::Mat>>& training_features,std::vector<cv::Mat>& features);
    
    // transform
    virtual void transform(cv::Mat& feature,unsigned int& id,double& weight,unsigned int* nid,int levelsup = 0);
    virtual void transform(cv::Mat& feature,unsigned int& id,double& weight);
    virtual void transform(cv::Mat& feature,unsigned int& id);
    
    // clustering utils
    void HK_means_step(unsigned int parent_id,std::vector<cv::Mat>& descriptors,int current_level);
    virtual void initiate_clusters(std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters);
    void initiate_clusters_KMpp(std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters);
    
    void create_words();
    void set_node_weights(std::vector<std::vector<cv::Mat>>& features);
    void load_fromtxt(std::string& file_name);

protected:
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