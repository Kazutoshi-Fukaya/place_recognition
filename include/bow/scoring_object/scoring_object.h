#ifndef SCORING_OBJECT_H_
#define SCORING_OBJECT_H_

#include <iostream>
#include <cfloat>

#include "bow/bow_vector/bow_vector.h"

namespace place_recognition
{
class GeneralScoring
{
public:
    GeneralScoring();
    GeneralScoring(std::string name);
    ~GeneralScoring();

    void set_params(std::string name,bool must_normalize,LNorm l_norm);
    double score(BowVector& v,BowVector& w);
    bool must_normalize(LNorm& l_norm);

private:
    double l1_score(BowVector& v,BowVector& w);
    double l2_score(BowVector& v,BowVector& w);
    double chi_square_score(BowVector& v,BowVector& w);
    double kl_score(BowVector& v,BowVector& w);
    double bhattacharyya_score(BowVector& v,BowVector& w);
    double dot_product_score(BowVector& v,BowVector& w);

    std::string NAME;
    bool MUST_NORMALIZE;
    LNorm L_NORM;
    double LOG_EPS_ = std::log(DBL_EPSILON);
};
}

#endif  // SCORING_OBJECT_H_
