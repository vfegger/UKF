#include "InputData.hpp"
#include <iostream>

InputData::InputData(double* data_input, unsigned length_input)
{
    length = length_input;
    if(length == 0u){
        data = NULL;
        return;
    }
    data_input = new double[length];
    return;
}

InputData::~InputData()
{
    length = 0u;
    if(data != NULL){
        delete[] data;
        data = NULL;
        std::cout << "Data is NULL\n";
    }
    return;
}

void InputData::print(){
    std::cout << "Test Class - Input Data\n";
}