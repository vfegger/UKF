#include "InputData.hpp"
#include <iostream>

InputData::InputData(double* data_input, unsigned length_input){
    length = length_input;
    if(length == 0u){
        data = NULL;
        std::cout << "Null pointer\n";
        return;
    }
    std::cout << "New pointer\n";
    data = new double[length];
    for(unsigned i = 0u; i < length; i++){
        data[i] = data_input[i];
    }
    return;
}

InputData::~InputData(){
    length = 0u;
    if(data != NULL){
        delete[] data;
        std::cout << "Data is NULL\n";
    }
    return;
}

void InputData::print(){
    std::cout << "Test Class - Input Data\n";
    if(data == NULL){
        std::cout << "Data is empty. Trying to access it will generate an error.\n";
        return;
    }
    for(unsigned i = 0; i < length; i++){
        std::cout << data[i] << "\t";
    }
    std::cout << "\n";
    return;
}