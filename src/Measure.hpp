#ifndef MEASURE_HEADER
#define MEASURE_HEADER

#include "Data.hpp"
#include "Point.hpp"
#include "PointCovariance.hpp"

class Measure
{
private:
    Point* point;
    Point* realPoint;
    PointCovariance* pointCovariance;
    unsigned length_data;
public:
    Measure(Data* dataReal_input, Data* data_input, Data* dataCovariance_input, unsigned length_input);
    ~Measure();
    unsigned GetStateLength();
    Point* GetPoint();
    Point* GetRealPoint();
    PointCovariance* GetPointCovariance();
};
#endif