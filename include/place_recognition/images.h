#ifndef IMAGES_H_
#define IMAGES_H_

#include "place_recognition/image.h"

namespace place_recognition
{
class Images
{
public:
    Images() :
        x(0.0), y(0.0), theta(0.0) {}

    Images(double _x,double _y,double _theta) :
        x(_x), y(_y), theta(_theta) {}

    void set_equ_image(Image _equ) { equ = _equ; }
    void set_rgb_image(Image _rgb) { rgb = _rgb; }

    Image equ;
    Image rgb;

    double x;
    double y;
    double theta;

private:
};
} // namespace place_recognition

#endif	// IMAGES_H_