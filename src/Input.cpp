#include "Input.hpp"
#include "Data.hpp"
#include "Parameters.hpp"
#include <new>

Input::Input(Data* inputData_input, unsigned inputDataLength_input, Parameters* inputParameters_input, unsigned inputParametersLength_input)
{
    inputParametersLength = inputParametersLength_input;
    inputDataLength = inputDataLength_input;
    inputParameters = new(std::nothrow) Parameters[inputParametersLength];
    inputData = new(std::nothrow) Data[inputDataLength];
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