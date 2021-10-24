#include "State.hpp"
#include <new>
#include <iostream>

State::State(Data* data_input, unsigned length_input){
    length_data = length_input;
    data = new(std::nothrow) Data[length_input];
    if(data == NULL && length_input != 0u){
        std::cout << "State is NULL with length = " << length_input << "\n";
    }
    unsigned acc = 0u;
    for(unsigned i = 0u; i < length; i++){
        acc += length_data;
        data[i] = Data(data_input[i]);
    }
    length = acc;
}

State::~State(){
    delete[] state;
    delete[] data;
}

void State::UpdateArrayFromData()
{
    for(unsigned i = 0, j = 0u; i < length_data; i++){
        unsigned auxLength = data[i].GetLength();
        for(unsigned k = 0; k < auxLength; k++,j++){
            state[j] = data[i][k];
        }
    }
}

void State::UpdateDataFromArray()
{
    for(unsigned i = 0, j = 0u; i < length_data; i++){
        unsigned auxLength = data[i].GetLength();
        for(unsigned k = 0; k < auxLength; k++,j++){
            data[i][k] = state[j];
        }
    }
}
