#ifndef POINT_HEADER
#define POINT_HEADER

#include "Data.hpp"

class Point
{
private:
    Data* data;
    double* state;
    unsigned length_data;
    unsigned length_state;
public:
    Point();
    Point(Data* data_input, unsigned length_input);
    Point(Point* point);
    Point(Point& point);
    ~Point();
    Point& operator=(const Point& point_input);
    void UpdateArrayFromData();
    void UpdateDataFromArray();
    unsigned GetLengthState();
    unsigned GetLengthData();
    Data* GetData();
    double* GetState();
};
#endif