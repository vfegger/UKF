#ifndef MEASURE_HEADER
#define MEASURE_HEADER

#include "Data.hpp"
#include "Point.hpp"
#include "PointCovariance.hpp"

class Measure
{
private:
    Point* point;
    PointCovariance* pointCovariance;
    unsigned length_data;
public:
    Measure(Data* data_input, Data* dataCovariance_input, unsigned length_input);
    ~Measure();
    unsigned GetStateLength();
    Point* GetPoint();
    PointCovariance* GetPointCovariance();
};
#endif