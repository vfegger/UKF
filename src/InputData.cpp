#include "InputData.hpp"
#include <string>
#include <iostream>
#include <new>

InputData::InputData(){
    name = "";
    length = 0u;
    data = NULL;
    return;
}

InputData::InputData(std::string name_input, double* data_input, unsigned length_input){
    name = name_input;
    length = length_input;
    if(length == 0u){
        data = NULL;
        std::cout << "Null pointer\n";
        return;
    }
    data = new(std::nothrow) double[length];
    if(data == NULL){
        std::cout << name << ":\n";
        std::cout << "\tError in allocation of memory of size :" << length*sizeof(double) << "\n";
        return;
    }
    for(unsigned i = 0u; i < length; i++){
        data[i] = data_input[i];
    }
    return;
}

InputData::~InputData(){
    length = 0u;
    if(data != NULL){
        delete[] data;
    }
    return;
}

void InputData::print(){
    std::cout << name << "\n\t";
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