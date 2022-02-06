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
    unsigned outputDataStateLength;
    unsigned outputDataMeasureLength;
    Parameters_Int* parameters_Int;
    Parameters_UInt* parameters_UInt;
    Parameters_FP* parameters_FP;
    Data* data_state;
    Data* data_state_covariance;
    Data* data_measure;
    Data* data_measure_covariance;

    void ExportConcat(Data* data, unsigned length, std::string &out, std::string &ext, std::string &delimiter);
    void ExportDivide(Data* data, unsigned length, std::string &out, std::string &ext, std::string &delimiter);

    void ExportConcat(Parameters_Int* parameters, unsigned length, std::string &out, std::string &ext, std::string &delimiter);
    void ExportDivide(Parameters_Int* parameters, unsigned length, std::string &out, std::string &ext, std::string &delimiter);
    void ExportConcat(Parameters_UInt* parameters, unsigned length, std::string &out, std::string &ext, std::string &delimiter);
    void ExportDivide(Parameters_UInt* parameters, unsigned length, std::string &out, std::string &ext, std::string &delimiter);
    void ExportConcat(Parameters_FP* parameters, unsigned length, std::string &out, std::string &ext, std::string &delimiter);
    void ExportDivide(Parameters_FP* parameters, unsigned length, std::string &out, std::string &ext, std::string &delimiter);
public:
    Output();
    ~Output();
    void SetOutput(Parameters_Int* parameters_Int_in,Parameters_UInt* parameters_UInt_in,Parameters_FP* parameters_FP_in, Data* data_state_in, Data* data_state_covariance_in, Data* data_measure_in, Data* data_measure_covariance_in, unsigned outputParametersLength_Int_in, unsigned outputDataStateLength_in, unsigned outputDataMeasureLength_in);
    virtual void Export(std::string &out, std::string &ext, std::string &delimiter, bool concatOutput = false);
};

#endif