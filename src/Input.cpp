#include "Input.hpp"
#include "Data.hpp"
#include "Parameters.hpp"
#include <new>
#include <iostream>

Input::Input(Data* inputData_input, unsigned inputDataLength_input, Parameters* inputParameters_input, unsigned inputParametersLength_input)
{
    inputParametersLength = inputParametersLength_input;
    inputDataLength = inputDataLength_input;
    inputParameters = new(std::nothrow) Parameters[inputParametersLength];
    inputData = new(std::nothrow) Data[inputDataLength];
    inputDataCovariance = new(std::nothrow) Data[inputDataLength];
    for(unsigned i = 0u; i < inputParametersLength; i++){
        inputParameters[i] = inputParameters_input[i];
    }
    for(unsigned i = 0u; i < inputDataLength; i++){
        inputData[i] = inputData_input[i];
    }
}

Input::~Input()
{
    inputParametersLength = 0u;
    inputDataLength = 0u;
    if(inputDataCovariance != NULL){
        delete[] inputDataCovariance;
    }
    if(inputData != NULL){
        delete[] inputData;
    }
    if(inputParameters != NULL){
        delete[] inputParameters;
    }
}

void Input::GetState(State &state_output){
    state_output = State(inputData, inputDataLength);
}

void Input::GetCovariance(double* &covariance_output){
    long unsigned acc = 0u;
    for(unsigned i = 0; i < inputDataLength; i++){
        acc += inputData[i].GetLength();
    }
    covariance_output = new(std::nothrow) double[acc*acc];
    for(unsigned i = 0; i < inputDataLength; i++){
        unsigned stateLength = inputData[i].GetLength();
        unsigned covarLength = inputDataCovariance[i].GetLength();
        if(covarLength == stateLength){
            for(unsigned j = 0u; j < covarLength; j++){
                covariance_output[j * acc + j] = inputDataCovariance[i][j];
            }
        } else {
            for(unsigned j = 0u; j < stateLength; j++){
                for(unsigned k = 0u; j < stateLength; j++){
                    covariance_output[j * acc + k] = inputDataCovariance[i][j* stateLength + k];
                }
            }
        }
    }
}

void Input::GetParameters(Parameters* &parameters_output){
    parameters_output = inputParameters;
}