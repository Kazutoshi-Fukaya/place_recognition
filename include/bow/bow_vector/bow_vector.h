#ifndef BOW_VECTOR_H_
#define BOW_VECTOR_H_

#include <map>
#include <vector>
#include <cmath>

#include "bow/bow_vector/l_norm.h"
#include "bow/bow_vector/weighting_type.h"
#include "bow/bow_vector/scoring_type.h"

namespace place_recognition
{
// Vector of words to represent images
// @param node_id       (unsigned int)
// @param word_value    (double)
class BowVector : public std::map<unsigned int,double>
{
public:
    BowVector();
    ~BowVector();

    void add_weight(unsigned int id,double value);
    void add_if_not_exist(unsigned int id,double value);
    void normalize(LNorm norm_type);
    uint16_t get_signature();

private:
};
}

#endif  // BOW_VECTOR_H_
