#ifndef MISMATCH_POINT_H_
#define MISMATCH_POINT_H_

namespace place_recognition
{
class MismatchPoint
{
public:
    MismatchPoint() :
        x(0.0), y(0.0), theta(0.0) {}
    MismatchPoint(double _x,double _y,double _theta) :
        x(_x), y(_y), theta(_theta) {}

    double x;
    double y;
    double theta;

private:
};
} // namespace place_recognition

#endif  // MISMATCH_POINT_H_
