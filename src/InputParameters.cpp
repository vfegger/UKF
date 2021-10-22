#include "InputParameters.hpp"
#include <iostream>
#include <new>

InputParameters::InputParameters(){
    name = "";
    length = 0u;
    parameters = NULL;
    return;
}

InputParameters::InputParameters(std::string name_input, double* parameters_input, unsigned length_input){
    name = name_input;
    length = length_input;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return;
    }
    parameters = new(std::nothrow) int[length];
    if(parameters == NULL){
        std::cout << name << ":\n";
        std::cout << "\tError in allocation of memory of size :" << length*sizeof(double) << "\n";
        return;
    }
    for(unsigned i = 0u; i < length; i++){
        parameters[i] = parameters_input[i];
    }
    return;
}

InputParameters::~InputParameters(){
    length = 0u;
    if(parameters != NULL){
        delete[] parameters;
    }
    return;
}

void InputParameters::print(){
    std::cout << "Test Class - Input Data\n";
    if(parameters == NULL){
        std::cout << "Data is empty. Trying to access it will generate an error.\n";
        return;
    }
    for(unsigned i = 0; i < length; i++){
        std::cout << parameters[i] << "\t";
    }
    std::cout << "\n";
    return;
}