#include "Input.hpp"
#include "InputData.hpp"
#include "InputParameters.hpp"
#include <new>

Input::Input(InputData* inputData_input, unsigned inputDataLength_input, InputParameters* inputParameters_input, unsigned inputParametersLength_input)
{
    inputParametersLength = inputParametersLength_input;
    inputDataLength = inputDataLength_input;
    inputParameters = new(std::nothrow) InputParameters[inputParametersLength];
    inputData = new(std::nothrow) InputData[inputDataLength];
    for(unsigned i = 0u; i < inputParametersLength; i++){
        inputParameters[i] = inputParameters_input[i];
    }
    for(unsigned i = 0u; i < inputDataLength; i++){
        inputData[i] = inputData_input[i];
    }
}

Input::~Input()
{
    delete[] inputData;
    delete[] inputParameters;
}