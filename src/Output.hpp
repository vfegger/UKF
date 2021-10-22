#ifndef OUTPUT_HEADER
#define OUTPUT_HEADER

#include "Data.hpp"
#include "Parameters.hpp"

class Output
{
private:
    Parameters* parameters;
    Data* data;
public:
    Output();
    ~Output();
};

#endif