#ifndef INPUT_HEADER
#define INPUT_HEADER

#include "InputData.hpp"
#include "InputParameters.hpp"

class Input
{
private:
    unsigned inputDataLength;
    unsigned inputParametersLength;
    InputParameters* inputParameters;
    InputData* inputData;
public:
    Input(InputData* inputData_input, unsigned inputDataLength_input, InputParameters* inputParameters_input, unsigned inputParametersLength_input);
    ~Input();
    virtual void Evolution(InputData* inputData_input, InputParameters* inputParameters_input) = 0;
    virtual void Observation(InputData* inputData_input, InputParameters* inputParameters_input) = 0;

};

#endif