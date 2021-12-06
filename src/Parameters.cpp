#include "Parameters.hpp"
#include <iostream>
#include <new>

Parameters::Parameters(){
    name = "";
    length = 0u;
    parameters = NULL;
    return;
}

Parameters::Parameters(std::string name_input, int* parameters_input, unsigned length_input){
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

Parameters::Parameters(Parameters& parameters_input){
    name = parameters_input.name;
    length = parameters_input.length;
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
        parameters[i] = parameters_input.parameters[i];
    }
    return;
}

Parameters::~Parameters(){
    length = 0u;
    if(parameters != NULL){
        delete[] parameters;
    }
    return;
}

Parameters& Parameters::operator=(const Parameters& parameters_input){
    name = parameters_input.name;
    length = parameters_input.length;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return *this;
    }
    parameters = new(std::nothrow) int[length];
    if(parameters == NULL){
        std::cout << name << ":\n";
        std::cout << "\tError in allocation of memory of size :" << length*sizeof(double) << "\n";
        return *this;
    }
    for(unsigned i = 0u; i < length; i++){
        parameters[i] = parameters_input.parameters[i];
    }
    return *this;
}

unsigned Parameters::GetLength(){
    return length;
}

std::string Parameters::GetName(){
    return name;
}

int& Parameters::operator[](unsigned index){
    return parameters[index];
}

const int& Parameters::operator[](unsigned index) const {
    return parameters[index];
}

void Parameters::print(){
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