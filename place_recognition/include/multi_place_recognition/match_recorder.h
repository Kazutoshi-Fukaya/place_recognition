#ifndef MATCH_RECORDER_H_
#define MATCH_RECORDER_H_

#include <iostream>
#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>

namespace place_recognition
{
class MatchPosition
{
public:
    MatchPosition() :
        x(0.0), y(0.0), theta(0.0) {}

    MatchPosition(double _x,double _y,double _theta) :
        x(_x), y(_y), theta(_theta) {}

    double x;
    double y;
    double theta;

private:
};

class MatchRecord
{
public:
    MatchRecord() {}
    MatchRecord(double t,MatchPosition est_pos,MatchPosition ref_pos) :
        time(t), est_position(est_pos), ref_position(ref_pos) {}

    double get_error()
    {
        double diff_x = est_position.x - ref_position.x;
        double diff_y = est_position.y - ref_position.y;
        return std::sqrt(diff_x*diff_x + diff_y*diff_y);
    }

    double time;
    MatchPosition est_position;
    MatchPosition ref_position;

private:
};

class MatchRecorder : public std::vector<MatchRecord>
{
public:
    MatchRecorder() {}

    void add_match_record(double t,MatchPosition est_pos,MatchPosition ref_pos)
    {
        this->emplace_back(MatchRecord(t,est_pos,ref_pos));
    }

    void set_error_th(double _error_th) { ERROR_TH = _error_th;}

private:
    double ERROR_TH = 1.5;
};
} // namespace place_recognition

#endif  // MATCH_RECORDER_H_
