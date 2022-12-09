#ifndef NODE_H_
#define NODE_H_

#include <vector>

#include <opencv2/core/core.hpp>

namespace place_recognition
{
class Node
{
public:
    Node() :
        id_(0), weight_(0.0), parent_(0), word_id_(0) {}

    Node(unsigned int id) :
        id_(id), weight_(0.0), parent_(0), word_id_(0) {}

    bool is_leaf() { return children_.empty(); }

    unsigned int id_;                       // node id
    double weight_;                         // Weight if the node is a word
    std::vector<unsigned int> children_;    // Children
    unsigned int parent_;                   // Parent node
    cv::Mat descriptors_;                   // Node descriptors
    unsigned int word_id_;                  // Word id if the node is a word

private:
};
}

#endif  // NODE_H_
