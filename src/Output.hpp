#ifndef OUTPUT_HEADER
#define OUTPUT_HEADER

#include "Data.hpp"
#include "Parameters.hpp"
#include <iostream>
#include <fstream>

class Output
{
private:
    unsigned outputDataStateLength;
    unsigned outputDataMeasureLength;
    Parameters* parameters;
    Data* data_state;
    Data* data_state_covariance;
    Data* data_measure;
    Data* data_measure_covariance;

    void ExportConcat(Data* data, unsigned length, std::string &out, std::string &ext, std::string &delimiter);
    void ExportDivide(Data* data, unsigned length, std::string &out, std::string &ext, std::string &delimiter);

    void ExportConcat(Parameters* parameters, std::string &out, std::string &ext, std::string &delimiter);
    void ExportDivide(Parameters* parameters, std::string &out, std::string &ext, std::string &delimiter);
public:
    Output();
    ~Output();
    void SetOutput(Parameters* parameters_Int_in, Data* data_state_in, Data* data_state_covariance_in, Data* data_measure_in, Data* data_measure_covariance_in, unsigned outputDataStateLength_in, unsigned outputDataMeasureLength_in);
    virtual void Export(std::string &out, std::string &ext, std::string &delimiter, bool concatOutput = false);
};

#endif