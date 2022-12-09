#include "bow/vocabulary/vocabulary.h"

using namespace place_recognition;

Vocabulary::Vocabulary() :
	descriptors_manipulator_(new DescriptorsManipulator()),
	m_k(10), m_L(5), m_weighting(TF_IDF), m_scoring(L1_NORM),
	m_scoring_object(NULL)
{
	create_scoring_object();
}

Vocabulary::Vocabulary(int k,int L,WeightingType weighting_type,ScoringType scoring_type) :
	descriptors_manipulator_(new DescriptorsManipulator()),
	m_k(k), m_L(L), m_weighting(weighting_type), m_scoring(scoring_type),
	m_scoring_object(NULL)
{
	create_scoring_object();
}

void Vocabulary::create(std::vector<cv::Mat>& training_features)
{
	// std::cout << "size: " << training_features.size() << std::endl;
	// for(const auto &tf : training_features){
		// std::cout << "(Cols,Rows): (" << tf.cols << "," << tf.rows << ")" << std::endl; 
	// }

	std::vector<std::vector<cv::Mat>> vtf(training_features.size());
	for(size_t i = 0; i < training_features.size(); i++){
		vtf[i].resize(training_features[i].rows);
		for(int r = 0; r < training_features[i].rows; r++){
			vtf[i][r] = training_features[i].rowRange(r,r+1);
			// std::cout << training_features[i].rowRange(r,r+1) << std::endl;
		}
	}
	create(vtf);
}

void Vocabulary::create(std::vector<std::vector<cv::Mat>>& training_features)
{
	// std::cout << "size: " << training_features.size() << std::endl;
	// for(const auto &vtf : training_features){
		// std::cout << vtf.size() << std::endl;
		// for(const auto &tf : vtf){
			// std::cout << "(Cols,Rows): (" << tf.cols << "," << tf.rows << ")" << std::endl;
		// }
	// }

	m_nodes.clear();
	m_words.clear();

	int expected_nodes = (int)((std::pow((double)m_k,(double)m_L + 1) - 1)/(m_k - 1));
	m_nodes.reserve(expected_nodes);

	std::vector<cv::Mat> features;
	get_features(training_features,features);

	// create root
	m_nodes.emplace_back(Node(0));

	// create the tree (Hierarchicak k-mean)
	HKmean_step(0,features,1);

	// create the words
  	create_words();

  	// and set the weight of each node of the tree
  	set_node_weights(training_features);
}

/*
void Vocabulary::transform(std::vector<cv::Mat>& features,BowVector& v)
{
	v.clear();
	if(empty()) return;
	
	// normalize
	LNorm norm;
	bool must = m_scoring_object->must_normalize(norm);
	
	if(m_weighting == TF || m_weighting == TF_IDF){
		for(auto fit = features.begin(); fit < features.end(); fit++){
			unsigned int id;
			double w;
			transform(*fit, id, w);
			
			// not stopped
			if(w > 0) v.add_weight(id,w);
    	}
		
		if(!v.empty() && !must){
			// unnecessary when normalizing
			const double nd = v.size();
			for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
    	}
	}
	else{
		for(auto fit = features.begin(); fit < features.end(); fit++){
			unsigned int id;
			double w;
			transform(*fit,id,w);

      		// not stopped
			if(w > 0) v.add_if_not_exist(id,w);

    	} // if add_features
  	} // if m_weighting == ...

  	if(must) v.normalize(norm);
}
*/

void Vocabulary::transform(cv::Mat& features,BowVector& v)
{
	v.clear();
    if(empty()) return;
    
    // normalize
    LNorm norm;
    bool must = m_scoring_object->must_normalize(norm);
    if(m_weighting == TF || m_weighting == TF_IDF){
		for(int r = 0; r < features.rows; r++){
			unsigned int id;
			double w;


         	// w is the idf value if TF_IDF, 1 if TF
			auto img = features.row(r);
			transform(img,id,w);	// TO DO
            // not stopped
            if(w > 0)  v.add_weight(id,w);
        }

        if(!v.empty() && !must){
            // unnecessary when normalizing
            const double nd = v.size();
            for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) vit->second /= nd;
        }

    }
    else{
		for(int r = 0; r < features.rows; r++){
			unsigned int id;
			double w;

            // w is idf if IDF, or 1 if BINARY
			auto img = features.row(r);
            transform(img,id,w);

            // not stopped
            if(w > 0) v.add_if_not_exist(id,w);

        } // if add_features
    } // if m_weighting == ...

    if(must) v.normalize(norm);
}

void Vocabulary::create_scoring_object()
{
	delete m_scoring_object;
	m_scoring_object = NULL;

	switch(m_scoring)
	{
	case L1_NORM:
		m_scoring_object = new GeneralScoring("L1Scoring");
		break;

	case L2_NORM:	
		m_scoring_object = new GeneralScoring("L2Scoring");
		break;
	
	case CHI_SQUARE:
		m_scoring_object = new GeneralScoring("ChiSquareScoring");
		break;
	
	case KL:
		m_scoring_object = new GeneralScoring("KLScoring");
		break;
	
	case BHATTACHARYYA:
		m_scoring_object = new GeneralScoring("BhattacharyyaScoring");
		break;

	case DOT_PRODUCT:
		m_scoring_object = new GeneralScoring("DotProductScoring");
		break;

	default:
		std::cerr << "No applicable mode." << std::endl;
		std::cerr << "Please select 'L1Scoring', 'L2Scoring', 'ChiSquareScoring', 'KLScoring', 'BhattacharyyaScoring' or 'DotProductScoring'" << std::endl;
		std::cerr << "Set 'L1Scoring'" << std::endl;
		m_scoring_object = new GeneralScoring("L1Scoring");
		break;
	}
}

void Vocabulary::get_features(std::vector<std::vector<cv::Mat>>& training_features,
	                          std::vector<cv::Mat>& features)
{
	features.resize(0);
	for(size_t i = 0; i < training_features.size(); i++){
		for(size_t j = 0; j < training_features[i].size(); j++){
			features.emplace_back(training_features[i][j]);
		}
	}
}

void Vocabulary::HKmean_step(unsigned int parent_id,std::vector<cv::Mat>& descriptors,int current_level)
{
	if(descriptors.empty()) return;

	// features associated to each cluster
	std::vector<cv::Mat> clusters;
	std::vector<std::vector<unsigned int>> groups;

	clusters.reserve(m_k);
	groups.reserve(m_k);

	if((int)descriptors.size() <= m_k){
		// one cluster per features
		groups.resize(descriptors.size());
		for(unsigned int i = 0; i < descriptors.size(); i++){
			groups[i].emplace_back(i);
			clusters.emplace_back(descriptors[i]);
		}
	}
	else{
		// select clusters and groups with kmeans
		bool first_time = true;
		bool goon = true;

		// to check if clusters move after iterations
		std::vector<int> last_association, current_association;
		while(goon){
			// 1. calculate cluster
			if(first_time){
				// random sample
				initiate_clusters(descriptors,clusters);
			}
			else{
				// calculate cluster center
				for(unsigned int c = 0; c < clusters.size(); c++){
					std::vector<cv::Mat> clusters_descriptors;
					clusters_descriptors.reserve(groups[c].size());
					// std::vector<unsigned int>::const_iterator vit;
					for(auto vit = groups[c].begin(); vit != groups[c].end(); vit++){
						clusters_descriptors.emplace_back(descriptors[*vit]);
					}
					descriptors_manipulator_->mean_value(clusters_descriptors,clusters[c]);
				}
			}

			// 2. Associate features with cluster
			groups.clear();
			groups.resize(clusters.size(),std::vector<unsigned int>());
			current_association.resize(descriptors.size());

			for(auto fit = descriptors.begin(); fit != descriptors.end(); fit++){
				double best_dist = descriptors_manipulator_->distance((*fit),clusters[0]);
                unsigned int icluster = 0;

                for(unsigned int c = 1; c < clusters.size(); c++){
                    double dist = descriptors_manipulator_->distance((*fit),clusters[c]);
                    if(dist < best_dist){
                        best_dist = dist;
                        icluster = c;
                    }
                }

                groups[icluster].emplace_back(fit - descriptors.begin());
                current_association[fit - descriptors.begin()] = icluster;
            }

            // kmeans++ ensures all the clusters has any feature associated with them
            // 3. check convergence
            if(first_time) first_time = false;
			else{
				goon = false;
                for(unsigned int i = 0; i < current_association.size(); i++){
                    if(current_association[i] != last_association[i]){
                        goon = true;
                        break;
                    }
                }
            }

            if(goon){
				// copy last feature-cluster association
                last_association = current_association;
            }
        } 
    }

    // create nodes
    for(unsigned int i = 0; i < clusters.size(); i++){
        unsigned int id = m_nodes.size();
        m_nodes.emplace_back(Node(id));
        m_nodes.back().descriptors_ = clusters[i];
        m_nodes.back().parent_ = parent_id;
        m_nodes[parent_id].children_.emplace_back(id);
    }

    // go on with the next level
    if(current_level < m_L){
        // iterate again with the resulting clusters
        const std::vector<unsigned int> &children_ids = m_nodes[parent_id].children_;
        for(unsigned int i = 0; i < clusters.size(); i++){
            unsigned int id = children_ids[i];
            std::vector<cv::Mat> child_features;
            child_features.reserve(groups[i].size());

            std::vector<unsigned int>::const_iterator vit;
            for(vit = groups[i].begin(); vit != groups[i].end(); vit++){
                child_features.push_back(descriptors[*vit]);
            }

            if(child_features.size() > 1){
                HKmean_step(id,child_features,current_level + 1);
            }
        }
    }
}

void Vocabulary::initiate_clusters(std::vector<cv::Mat>& descriptors,std::vector<cv::Mat>& clusters)
{
	initiate_clusters_KMpp(descriptors,clusters);
}

void Vocabulary::initiate_clusters_KMpp(std::vector<cv::Mat>& pfeatures,std::vector<cv::Mat>& clusters)
{
	// Implements kmeans++ seeding algorithm

	clusters.resize(0);
	clusters.reserve(m_k);
	std::vector<double> min_dists(pfeatures.size(),std::numeric_limits<double>::max());

	// 1. Choose one center uniformly at random from among the data points.
	int ifeature = std::rand()%pfeatures.size();
	
	// create first cluster
	clusters.emplace_back(pfeatures[ifeature]);

	// compute the initial distances
	std::vector<double>::iterator dit;
	dit = min_dists.begin();
	for(auto fit = pfeatures.begin(); fit != pfeatures.end(); fit++, dit++){
		*dit = descriptors_manipulator_->distance((*fit),clusters.back());
	}

	while((int)clusters.size() < m_k){
		// 2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
		dit = min_dists.begin();
		for(auto fit = pfeatures.begin(); fit != pfeatures.end(); fit++, dit++){
			if(*dit < 0){
				double dist = descriptors_manipulator_->distance((*fit),clusters.back());
				if(dist < *dit) *dit = dist;
			}
		}

		// 3. Add one new data point as a center.
		//    Each point x is chosen with probability promotional to D(x)^2 
		double dist_sum = std::accumulate(min_dists.begin(),min_dists.end(),0.0);
		if(dist_sum > 0){
			double cut_d;
			do{
				cut_d = (double(std::rand())/double(RAND_MAX))*dist_sum;
			}while(cut_d == 0);
			
			double d_up_now = 0;
			for(dit = min_dists.begin(); dit != min_dists.end(); dit++){
				d_up_now += *dit;
				if(d_up_now >= cut_d) break;
      		}
			if(dit == min_dists.end()) ifeature = pfeatures.size()-1;
			else ifeature = dit - min_dists.begin();
			
			clusters.emplace_back(pfeatures[ifeature]);
    	} 
		else break;
	}
}

void Vocabulary::create_words()
{
	m_words.resize(0);
	if(!m_nodes.empty()){
		m_words.reserve((int)std::pow((double)m_k,(double)m_L));
		
		auto  nit = m_nodes.begin(); // ignore root
		for(nit++; nit != m_nodes.end(); nit++){
			if(nit->is_leaf()){
				nit->word_id_ = m_words.size();
				m_words.emplace_back(&(*nit));
			}
		}
	}
}

void Vocabulary::set_node_weights(std::vector<std::vector<cv::Mat>>& training_features)
{
	const unsigned int NWords = m_words.size();
	const unsigned int NDocs = training_features.size();
	
	if(m_weighting == TF || m_weighting == BINARY){
		// idf part must be 1 always
		for(unsigned int i = 0; i < NWords; i++) m_words[i]->weight_ = 1;
  	}
	else if(m_weighting == IDF || m_weighting == TF_IDF){
		// IDF and TF-IDF: we calculte the idf path now

    	// Note: this actually calculates the idf part of the tf-idf score.
    	// The complete tf-idf score is calculated in ::transform

    	std::vector<unsigned int> Ni(NWords,0);
    	std::vector<bool> counted(NWords,false);
		for(auto mit = training_features.begin(); mit != training_features.end(); mit++){
			std::fill(counted.begin(), counted.end(), false);
			
			for(auto fit = mit->begin(); fit < mit->end(); fit++){
				unsigned int word_id;
				transform(*fit,word_id);

        		if(!counted[word_id]){
					Ni[word_id]++;
					counted[word_id] = true;
        		}
      		}
    	}
		
		// set ln(N/Ni)
		for(unsigned int i = 0; i < NWords; i++){
			if(Ni[i] > 0) m_words[i]->weight_ = std::log((double)NDocs/(double)Ni[i]);
      	}
    }
}

void Vocabulary::transform(cv::Mat& feature,unsigned int& id)
{
	double weight;
	transform(feature,id,weight);
}

void Vocabulary::transform(cv::Mat& feature,unsigned int& word_id,double& weight)
{
	unsigned int final_id = 0;	// root
	
	//binary descriptor
	if(feature.type() == CV_8U){
		do{
			auto const &nodes = m_nodes[final_id].children_;
			uint64_t best_d = std::numeric_limits<uint64_t>::max();
			int idx = 0, bestidx = 0;
			for(const auto &id : nodes){
				//compute distance
				uint64_t dist= descriptors_manipulator_->distance_8uc1(feature,m_nodes[id].descriptors_);
				if(dist < best_d){
					best_d = dist;
                   	final_id = id;
                  	bestidx = idx;
				}
				idx++;
			}
		}while(!m_nodes[final_id].is_leaf());
	}
	else{
		do{
			auto const  &nodes = m_nodes[final_id].children_;
		  	uint64_t best_d = std::numeric_limits<uint64_t>::max();
		  	int idx = 0, bestidx = 0;
		  	for(const auto &id : nodes){
				//compute distance
				uint64_t dist = descriptors_manipulator_->distance(feature,m_nodes[id].descriptors_);
				if(dist < best_d){
					best_d = dist;
					final_id = id;
					bestidx = idx;
				}
				idx++;
			}	
		} while (!m_nodes[final_id].is_leaf());
  	}
	
	// turn node id into word id
  	word_id = m_nodes[final_id].word_id_;
  	weight = m_nodes[final_id].weight_;
}

int Vocabulary::get_branching_factor() { return m_k; }

int Vocabulary::get_depth_levels() { return m_L; }

WeightingType Vocabulary::get_weighting_type() { return m_weighting; }

ScoringType Vocabulary::get_scoring_type() { return m_scoring; }

unsigned int Vocabulary::size() { return (unsigned int)m_words.size(); }

bool Vocabulary::empty() { return m_words.empty(); }

// for debug
void Vocabulary::get_info()
{
	std::cout << "k = " << get_branching_factor() << std::endl;
	std::cout << "L = " << get_depth_levels() << std::endl;
	
	std::cout << "Weighting = ";
	switch(get_weighting_type())
	{
		case TF_IDF: 
			std::cout << "tf-idf" << std::endl;
			break;

    	case TF: 
			std::cout << "tf" << std::endl; 
			break;

    	case IDF: 
			std::cout << "idf" << std::endl;
			break;
		
		case BINARY: 
			std::cout << "binary" << std::endl;
			break;
  	}

	std::cout << "Scoring = ";
	switch(get_scoring_type())
	{
		case L1_NORM: 
			std::cout << "L1-norm" << std::endl;
			break;
		
		case L2_NORM: 
			std::cout << "L2-norm" << std::endl;
			break;
		
		case CHI_SQUARE: 
			std::cout << "Chi square distance" << std::endl;
			break;
		
		case KL: 
			std::cout << "KL-divergence" << std::endl;
			break;
		
		case BHATTACHARYYA: 
			std::cout << "Bhattacharyya coefficient" << std::endl;
			break;
		
		case DOT_PRODUCT: 
			std::cout << "Dot product" << std::endl;
			break;
  	}
	
	std::cout << "Number of words = " << size() << std::endl;
	std::cout << std::endl;
}

double Vocabulary::score(BowVector& a,BowVector& b)
{
	return m_scoring_object->score(a,b);
}