#ifndef INPUT_HEADER
#define INPUT_HEADER

#include "Data.hpp"
#include "Parameters.hpp"

class Input
{
private:
    unsigned inputDataLength;
    unsigned inputParametersLength;
    Parameters* inputParameters;
    Data* inputData;
public:
    Input(Data* inputData_input, unsigned inputDataLength_input, Parameters* inputParameters_input, unsigned inputParametersLength_input);
    ~Input();
    virtual void Evolution(Data* inputData_input, Parameters* inputParameters_input) = 0;
    virtual void Observation(Data* inputData_input, Parameters* inputParameters_input) = 0;

};

#endif