#ifndef STATE_HEADER
#define STATE_HEADER

#include "Data.hpp"
#include "Point.hpp"
#include "PointCovariance.hpp"

class State
{
private:
    Point* point;
    PointCovariance* pointCovariance;
    unsigned length_data;
public:
    State(Data* data_input, Data* dataCovariance_input, unsigned length_input);
    ~State();
    unsigned GetStateLength();
    Point* GetPoint();
    PointCovariance* GetPointCovariance();
};
#endif