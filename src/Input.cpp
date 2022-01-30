#include "Input.hpp"
#include "Data.hpp"
#include "Parameters.hpp"
#include <new>
#include <iostream>

Input::Input(){
    inputParametersLength = 0u;
    inputDataLength = 0u;
    measureDataLength = 0u;
    inputParameters = NULL;
    inputData = NULL;
    inputDataCovariance = NULL;
    measureData = NULL;
    measureDataNoise = NULL;
}

void Input::Initialize(Data* inputData_input, Data* inputDataCovariance_input, Data* inputDataNoise_input, unsigned inputDataLength_input, Parameters* inputParameters_input, unsigned inputParametersLength_input, Data* measureData_input, Data* measureDataNoise_input, unsigned measureDataLength_input){
    inputParametersLength = inputParametersLength_input;
    inputDataLength = inputDataLength_input;
    measureDataLength = measureDataLength_input;
    inputParameters = new(std::nothrow) Parameters[inputParametersLength];
    inputData = new(std::nothrow) Data[inputDataLength];
    inputDataCovariance = new(std::nothrow) Data[inputDataLength];
    inputDataNoise = new(std::nothrow) Data[inputDataLength];
    measureData = new(std::nothrow) Data[measureDataLength];
    measureDataNoise = new(std::nothrow) Data[measureDataLength];
    for(unsigned i = 0u; i < inputParametersLength; i++){
        inputParameters[i] = Parameters(inputParameters_input[i]);
    }
    for(unsigned i = 0u; i < inputDataLength; i++){
        inputData[i] = Data(inputData_input[i]);
    }
    for(unsigned i = 0u; i < inputDataLength; i++){
        inputDataCovariance[i] = Data(inputDataCovariance_input[i]);
    }
    for(unsigned i = 0u; i < inputDataLength; i++){
        inputDataNoise[i] = Data(inputDataNoise_input[i]);
    }
    for(unsigned i = 0u; i < measureDataLength; i++){
        measureData[i] = Data(measureData_input[i]);
    }
    for(unsigned i = 0u; i < measureDataLength; i++){
        measureDataNoise[i] = Data(measureDataNoise_input[i]);
    }
}

Input::~Input()
{
    inputParametersLength = 0u;
    inputDataLength = 0u;
    measureDataLength = 0u;
    if(inputDataCovariance != NULL){
        delete[] inputDataCovariance;
    }
    if(inputData != NULL){
        delete[] inputData;
    }
    if(inputDataNoise != NULL){
        delete[] inputDataNoise;
    }
    if(inputParameters != NULL){
        delete[] inputParameters;
    }
    if(measureData != NULL){
        delete[] measureData;
    }
    if(measureDataNoise != NULL){
        delete[] measureDataNoise;
    }
}

void Input::GetState(State* state_output){
    state_output = new State(inputData, inputDataCovariance, inputDataLength);
}

State* Input::GetState(){
    return new State(inputData, inputDataCovariance, inputDataLength);
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

double* Input::GetCovariance(){
    long unsigned acc = 0u;
    for(unsigned i = 0; i < inputDataLength; i++){
        acc += inputData[i].GetLength();
    }
    double* covariance_output = new(std::nothrow) double[acc*acc];
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
    return covariance_output;
}

void Input::GetParameters(Parameters* &parameters_output){
    parameters_output = inputParameters;
}

Parameters* Input::GetParameters(){
    return inputParameters;
}

unsigned Input::GetParametersLength(){
    return inputParametersLength;
}

void Input::GetMeasure(Measure* &measure_output){
    measure_output = new Measure(measureData,measureDataNoise,measureDataLength);
}

Measure* Input::GetMeasure(){
    return new Measure(measureData,measureDataNoise,measureDataLength);
}
