#include "bow/vocabulary/vocabulary.h"

using namespace place_recognition;

Vocabulary::Vocabulary() :
	descriptors_manipulator_(new DescriptorsManipulator()),
	quick_lz_(new QuickLZ),
	m_k(10), m_L(5), m_weighting(TF_IDF), m_scoring(L1_NORM),
	m_scoring_object(NULL)
{
	create_scoring_object();
}

Vocabulary::Vocabulary(int k,int L,WeightingType weighting_type,ScoringType scoring_type) :
	descriptors_manipulator_(new DescriptorsManipulator()),
	quick_lz_(new QuickLZ),
	m_k(k), m_L(L), m_weighting(weighting_type), m_scoring(scoring_type),
	m_scoring_object(NULL)
{
	create_scoring_object();
}

Vocabulary::Vocabulary(std::string& file_name) :
	descriptors_manipulator_(new DescriptorsManipulator),
	quick_lz_(new QuickLZ),
	m_scoring_object(NULL)
{
	load(file_name);
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

void Vocabulary::save(std::string& file_name,bool binary_compressed)
{
	if(file_name.find(".yml") == std::string::npos){
        std::ofstream file_out(file_name,std::ios::binary);
        if(!file_out) throw std::runtime_error("Vocabulary::saveBinary Could not open file : "+file_name+" for writing");
		// TO DO

		// toStream(file_out,binary_compressed);
    }
    else{
        cv::FileStorage fs(file_name.c_str(),cv::FileStorage::WRITE);
		if(!fs.isOpened()) throw std::string("Could not open file ") + file_name;
        std::cout << file_name << std::endl;
		std::string name = "vocabulary";
		save(fs,name);
    }
}

void Vocabulary::save(cv::FileStorage& fs,std::string& name)
{
	fs << name << "{";
	fs << "k" << m_k;
	fs << "L" << m_L;
	fs << "scoringType" << m_scoring;
	fs << "weightingType" << m_weighting;
	
	// tree
	fs << "nodes" << "[";
	std::vector<unsigned int> parents, children;
	std::vector<unsigned int>::const_iterator pit;
	
	// root
	parents.emplace_back(0); 
	
	// node
	while(!parents.empty()){
		unsigned int pid = parents.back();
		parents.pop_back();
		
		Node& parent = m_nodes[pid];
		children = parent.children_;
		for(pit = children.begin(); pit != children.end(); pit++){
			Node& child = m_nodes[*pit];
			std::cout << m_nodes[*pit].id_ << " ";
			
			// save node data
			fs << "{:";
			fs << "nodeId" << (int)child.id_;
			fs << "parentId" << (int)pid;
			fs << "weight" << (double)child.weight_;
			fs << "descriptor" << descriptors_manipulator_->to_string(child.descriptors_);
			fs << "}";
			
			// add to parent list
			if(!child.is_leaf()) parents.emplace_back(*pit);
      
    	}
  	}
	std::cout<<"\n";
	
	fs << "]";
	
	// words
	fs << "words" << "[";
	for(auto wit = m_words.begin(); wit != m_words.end(); wit++){
		unsigned int id = wit - m_words.begin();
		fs << "{:";
		fs << "wordId" << (int)id;
		fs << "nodeId" << (int)(*wit)->id_;
		fs << "}";
  	}
	fs << "]"; 
	fs << "}";

	// fs.release();
}

void Vocabulary::to_stream(std::ostream& out_str,bool compressed)
{
	uint64_t sig = 88877711233;	// magic number describing the file
    out_str.write((char*)&sig,sizeof(sig));
    out_str.write((char*)&compressed,sizeof(compressed));
    uint32_t nnodes = m_nodes.size();
    out_str.write((char*)&nnodes,sizeof(nnodes));
    if(nnodes==0) return;

    // save everything to a stream
    std::stringstream aux_stream;
    aux_stream.write((char*)&m_k,sizeof(m_k));
    aux_stream.write((char*)&m_L,sizeof(m_L));
    aux_stream.write((char*)&m_scoring,sizeof(m_scoring));
    aux_stream.write((char*)&m_weighting,sizeof(m_weighting));
    
	// nodes
    std::vector<unsigned int> parents = {0};	// root
    while(!parents.empty()){
		unsigned int pid = parents.back();
        parents.pop_back();
        Node& parent = m_nodes[pid];
        for(auto pit : parent.children_){
			Node& child = m_nodes[pit];
            aux_stream.write((char*)&child.id_,sizeof(child.id_));
            aux_stream.write((char*)&pid,sizeof(pid));
            aux_stream.write((char*)&child.weight_,sizeof(child.weight_));
			descriptors_manipulator_->to_stream(child.descriptors_,aux_stream);

			// add to parent list
            if(!child.is_leaf()) parents.push_back(pit);
        }
    }

    // words
    // save size
    uint32_t m_words_size = m_words.size();
    aux_stream.write((char*)&m_words_size,sizeof(m_words_size));
    for(auto wit = m_words.begin(); wit != m_words.end(); wit++){
		unsigned int id = wit - m_words.begin();
        aux_stream.write((char*)&id,sizeof(id));
        aux_stream.write((char*)&(*wit)->id_,sizeof((*wit)->id_));
    }

	/*
    // now, decide if compress or not
    if(compressed){
		qlz_state_compress  state_compress;
        memset(&state_compress, 0, sizeof(qlz_state_compress));
        //Create output buffer
        int chunkSize=10000;
        std::vector<char> compressed( chunkSize+size_t(400), 0);
        std::vector<char> input( chunkSize, 0);
        int64_t total_size= static_cast<int64_t>(aux_stream.tellp());
        uint64_t total_compress_size=0;
        //calculate how many chunks will be written
        uint32_t nChunks= total_size / chunkSize;
        if ( total_size%chunkSize!=0) nChunks++;
        out_str.write((char*)&nChunks, sizeof(nChunks));
        //start compressing the chunks
		while (total_size != 0){
            int readSize=chunkSize;
            if (total_size<chunkSize) readSize=total_size;
            aux_stream.read(&input[0],readSize);
            uint64_t  compressed_size   = qlz_compress(&input[0], &compressed[0], readSize, &state_compress);
            total_size-=readSize;
            out_str.write(&compressed[0], compressed_size);
            total_compress_size+=compressed_size;
        }
    }
    else{
        out_str<<aux_stream.rdbuf();
    }
	*/
}

void Vocabulary::load(std::string& file_name)
{
	// check first if it is a binary file
    std::ifstream ifs(file_name,std::ios::binary);
    if (!ifs) throw std::runtime_error("Vocabulary::load Could not open file : "+file_name+" for reading");
    
	if(!load(ifs)){
		if(file_name.find(".txt") != std::string::npos) load_fromtxt(file_name);
	}
	else{
		cv::FileStorage fs(file_name.c_str(),cv::FileStorage::READ);
	    if(!fs.isOpened()) throw std::string("Could not open file ") + file_name;
		std::string name = "vocabulary";
		load(fs,name);
	}
}

bool Vocabulary::load(std::istream& ist)
{
	// magic number describing the file
	uint64_t sig;
    ist.read((char*)&sig,sizeof(sig));

	// Check if it is a binary file.
    if(sig != 88877711233) return false;
    ist.seekg(0,std::ios::beg);
    from_stream(ist);

    return true;
}

void Vocabulary::from_stream(std::istream& str)
{
	m_words.clear();
    m_nodes.clear();
    
	// magic number describing the file
	uint64_t sig = 0;
    str.read((char*)&sig,sizeof(sig));
    if(sig != 88877711233) throw std::runtime_error("Vocabulary::fromStream  is not of appropriate type");
    
	bool compressed;
    str.read((char*)&compressed,sizeof(compressed));
    
	uint32_t nnodes;
    str.read((char*)&nnodes,sizeof(nnodes));
    if(nnodes == 0) return;
    
	std::stringstream decompressed_stream;
    std::istream* _used_str = 0;
    if(compressed){
        QlzStateDecompress state_decompress;
        memset(&state_decompress,0,sizeof(QlzStateDecompress));
        int chunk_size = 10000;
        std::vector<char> decompressed(chunk_size);
        std::vector<char> input(chunk_size + 400);

        // read how many chunks are there
        uint32_t nChunks;
        str.read((char*)&nChunks,sizeof(nChunks));
        for(int i = 0; i < nChunks; i++){
            str.read(&input[0],9);
			int c = quick_lz_->qlz_size_compressed(&input[0]);
            str.read(&input[9],c - 9);
			size_t d = quick_lz_->qlz_decompress(&input[0],&decompressed[0],&state_decompress);
            decompressed_stream.write(&decompressed[0],d);
        }
        _used_str=&decompressed_stream;
    }
    else{
        _used_str=&str;
    }

    _used_str->read((char*)&m_k,sizeof(m_k));
    _used_str->read((char*)&m_L,sizeof(m_L));
    _used_str->read((char*)&m_scoring,sizeof(m_scoring));
    _used_str->read((char*)&m_weighting,sizeof(m_weighting));

	create_scoring_object();
    m_nodes.resize(nnodes );
    m_nodes[0].id_ = 0;

    for(size_t i = 1; i < m_nodes.size(); i++){
		unsigned int nid;
        _used_str->read((char*)&nid,sizeof(unsigned int));
        Node& child = m_nodes[nid];
        child.id_ = nid;
        _used_str->read((char*)&child.parent_,sizeof(child.parent_));
        _used_str->read((char*)&child.weight_,sizeof(child.weight_));
		descriptors_manipulator_->from_stream(child.descriptors_,*_used_str);
        m_nodes[child.parent_].children_.emplace_back(child.id_);
    }
	
	// words
    uint32_t m_words_size;
    _used_str->read((char*)&m_words_size,sizeof(m_words_size));
    m_words.resize(m_words_size);
    for(unsigned int i = 0; i < m_words.size(); i++){
        unsigned int wid;
		unsigned int nid;
        _used_str->read((char*)&wid,sizeof(wid));
        _used_str->read((char*)&nid,sizeof(nid));
        m_nodes[nid].word_id_ = wid;
        m_words[wid] = &m_nodes[nid];
    }
}

void Vocabulary::load_fromtxt(std::string& file_name)
{
	std::ifstream ifs(file_name);
    if(!ifs) throw std::runtime_error("Vocabulary:: load_fromtxt  Could not open file for reading: " + file_name);
    int n1, n2;

	{
		std::string str;
		getline(ifs,str);
		std::stringstream ss(str);
		ss >> m_k >> m_L >> n1 >> n2;
    }

    if(m_k < 0 || m_k > 20 || m_L < 1 || m_L > 10 || n1 < 0 || n1 > 5 || n2 < 0 || n2 > 3){
		throw std::runtime_error("Vocabulary loading failure: This is not a correct text file!");
	}
    m_scoring = (ScoringType)n1;
    m_weighting = (WeightingType)n2;
	create_scoring_object();

    // nodes
	int expected_nodes = (int)((std::pow((double)m_k,(double)m_L + 1) - 1)/(m_k - 1));
	m_nodes.reserve(expected_nodes);	
	m_words.reserve(pow((double)m_k, (double)m_L + 1));
	m_nodes.resize(1);
	m_nodes[0].id_ = 0;
	
	int counter = 0;
    while(!ifs.eof()){
		std::string snode;
		getline(ifs,snode);
		if (counter++ % 100 == 0) std::cerr << ".";
		if (snode.size() == 0) break;

		std::stringstream ssnode(snode);
		int nid = m_nodes.size();
        m_nodes.resize(m_nodes.size() + 1);
		m_nodes[nid].id_ = nid;
		
		int pid ;
		ssnode >> pid;
		m_nodes[nid].parent_ = pid;
		m_nodes[pid].children_.emplace_back(nid);
		
		int nIsLeaf;
		ssnode >> nIsLeaf;

        // read until the end and add to data
		std::vector<float> data;
		data.reserve(100);
		
		float d;
		while(ssnode >> d) data.emplace_back(d);
        
		// the weight is the last
		m_nodes[nid].weight_ = data.back();
		data.pop_back();	// remove
        
		// the rest, to the descriptor
		m_nodes[nid].descriptors_.create(1,data.size(),CV_8UC1);
		auto ptr=m_nodes[nid].descriptors_.ptr<uchar>(0);
		for(auto d : data) *ptr++ = d;
		
		if(nIsLeaf > 0){
			int wid = m_words.size();
			m_words.resize(wid + 1);
			m_nodes[nid].word_id_ = wid;
            m_words[wid] = &m_nodes[nid];
        }
        else m_nodes[nid].children_.reserve(m_k);
    }
}

void Vocabulary::load(cv::FileStorage& fs,std::string& name)
{
	m_words.clear();
	m_nodes.clear();

	cv::FileNode fvoc = fs[name];
	m_k = (int)fvoc["k"];
	m_L = (int)fvoc["L"];
	m_scoring = (ScoringType)((int)fvoc["scoringType"]);
	m_weighting = (WeightingType)((int)fvoc["weightingType"]);

	create_scoring_object();
	
	// nodes
	cv::FileNode fn = fvoc["nodes"];
	m_nodes.resize(fn.size() + 1);	// +1 to include root
	m_nodes[0].id_ = 0;
	
	for(unsigned int i = 0; i < fn.size(); i++){
		unsigned int nid = (int)fn[i]["nodeId"];
		unsigned int pid = (int)fn[i]["parentId"];
		double weight = (double)fn[i]["weight"];
		std::string d = (std::string)fn[i]["descriptor"];
		
		m_nodes[nid].id_ = nid;
		m_nodes[nid].parent_ = pid;
		m_nodes[nid].weight_ = weight;
		m_nodes[pid].children_.emplace_back(nid);
		descriptors_manipulator_->from_string(m_nodes[nid].descriptors_,d);
  	}
	
	// word
	fn = fvoc["words"];
	m_words.resize(fn.size());
	for(unsigned int i = 0; i < fn.size(); i++){
		unsigned int wid = (int)fn[i]["wordId"];
		unsigned int nid = (int)fn[i]["nodeId"];
		
		m_nodes[nid].word_id_ = wid;
		m_words[wid] = &m_nodes[nid];
  	}
}