#ifndef OUTPUT_HEADER
#define OUTPUT_HEADER

#include "Data.hpp"
#include "Parameters.hpp"
#include <iostream>
#include <fstream>

class Output
{
private:
    unsigned outputParametersLength;
    unsigned outputDataLength;
    Parameters* parameters;
    Data* data;
public:
    Output();
    ~Output();
    virtual void Export(std::string &out, std::string &ext, std::string &delimiter, bool concatOutput = false);
};

#endif