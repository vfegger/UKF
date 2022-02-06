#include "Parameters.hpp"
#include <iostream>
#include <new>

// Parameters_Int

Parameters_Int::Parameters_Int(){
    name = "";
    length = 0u;
    parameters = NULL;
    return;
}

Parameters_Int::Parameters_Int(std::string name_input, long long int* parameters_input, unsigned length_input){
    name = name_input;
    length = length_input;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return;
    }
    parameters = new(std::nothrow) long long int[length];
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

Parameters_Int::Parameters_Int(const Parameters_Int& parameters_input){
    name = parameters_input.name;
    length = parameters_input.length;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return;
    }
    parameters = new(std::nothrow) long long int[length];
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

Parameters_Int::~Parameters_Int(){
    length = 0u;
    if(parameters != NULL){
        delete[] parameters;
    }
    return;
}

Parameters_Int& Parameters_Int::operator=(const Parameters_Int& parameters_input){
    name = parameters_input.name;
    length = parameters_input.length;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return *this;
    }
    parameters = new(std::nothrow) long long int[length];
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

unsigned Parameters_Int::GetLength(){
    return length;
}

std::string Parameters_Int::GetName(){
    return name;
}

long long int& Parameters_Int::operator[](unsigned index){
    return parameters[index];
}

const long long int& Parameters_Int::operator[](unsigned index) const{
    return parameters[index];
}

void Parameters_Int::print(){
    std::cout << "Parameter Int Class - Input Data\n";
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

// Parameters_UInt

Parameters_UInt::Parameters_UInt(){
    name = "";
    length = 0u;
    parameters = NULL;
    return;
}

Parameters_UInt::Parameters_UInt(std::string name_input, long long unsigned* parameters_input, unsigned length_input){
    name = name_input;
    length = length_input;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return;
    }
    parameters = new(std::nothrow) long long unsigned[length];
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

Parameters_UInt::Parameters_UInt(const Parameters_UInt& parameters_input){
    name = parameters_input.name;
    length = parameters_input.length;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return;
    }
    parameters = new(std::nothrow) long long unsigned[length];
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

Parameters_UInt::~Parameters_UInt(){
    length = 0u;
    if(parameters != NULL){
        delete[] parameters;
    }
    return;
}

Parameters_UInt& Parameters_UInt::operator=(const Parameters_UInt& parameters_input){
    name = parameters_input.name;
    length = parameters_input.length;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return *this;
    }
    parameters = new(std::nothrow) long long unsigned[length];
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

unsigned Parameters_UInt::GetLength(){
    return length;
}

std::string Parameters_UInt::GetName(){
    return name;
}

long long unsigned& Parameters_UInt::operator[](unsigned index){
    return parameters[index];
}

const long long unsigned& Parameters_UInt::operator[](unsigned index) const{
    return parameters[index];
}

void Parameters_UInt::print(){
    std::cout << "Parameter UInt Class - Input Data\n";
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

// Parameters_FP

Parameters_FP::Parameters_FP(){
    name = "";
    length = 0u;
    parameters = NULL;
    return;
}

Parameters_FP::Parameters_FP(std::string name_input, double* parameters_input, unsigned length_input){
    name = name_input;
    length = length_input;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return;
    }
    parameters = new(std::nothrow) double[length];
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

Parameters_FP::Parameters_FP(const Parameters_FP& parameters_input){
    name = parameters_input.name;
    length = parameters_input.length;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return;
    }
    parameters = new(std::nothrow) double[length];
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

Parameters_FP::~Parameters_FP(){
    length = 0u;
    if(parameters != NULL){
        delete[] parameters;
    }
    return;
}

Parameters_FP& Parameters_FP::operator=(const Parameters_FP& parameters_input){
    name = parameters_input.name;
    length = parameters_input.length;
    if(length == 0u){
        parameters = NULL;
        std::cout << "Null pointer\n";
        return *this;
    }
    parameters = new(std::nothrow) double[length];
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

unsigned Parameters_FP::GetLength(){
    return length;
}

std::string Parameters_FP::GetName(){
    return name;
}

double& Parameters_FP::operator[](unsigned index){
    return parameters[index];
}

const double& Parameters_FP::operator[](unsigned index) const{
    return parameters[index];
}

void Parameters_FP::print(){
    std::cout << "Parameter Int Class - Input Data\n";
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

// Parameters

Parameters::Parameters(){
    Int_Length = 0;
    UInt_Length = 0;
    FP_Length = 0;
    Int = NULL;
    UInt = NULL;
    FP = NULL;
}

Parameters::Parameters(Parameters_Int* pointer_int, unsigned int_length, Parameters_UInt* pointer_uint, unsigned uint_length, Parameters_FP* pointer_fp, unsigned fp_length){
    Int_Length = int_length;
    UInt_Length = uint_length;
    FP_Length = fp_length;
    Int = new Parameters_Int[Int_Length];
    UInt = new Parameters_UInt[UInt_Length];
    FP = new Parameters_FP[FP_Length];
}

Parameters::~Parameters(){
    Int_Length = 0;
    UInt_Length = 0;
    FP_Length = 0;
    delete[] Int;
    delete[] UInt;
    delete[] FP;
}