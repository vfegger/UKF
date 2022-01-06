#include "PointCovariance.hpp"
#include <new>

PointCovariance::PointCovariance(){
    dataCovariance = NULL;
    stateCovariance = NULL;
    compactForm = NULL;
    length_data = 0u;
    length_state = 0u;
}

PointCovariance::PointCovariance(Data* data_input, Data* dataCovariance_input, unsigned length_input){
    length_data = length_input;
    dataCovariance = new(std::nothrow) Data[length_data];
    compactForm = new(std::nothrow) bool[length_data];
    unsigned acc = 0u;
    for(unsigned i = 0u; i < length_data; i++){
        unsigned aux1 = data_input[i].GetLength();
        unsigned aux2 = dataCovariance_input[i].GetLength();
        compactForm[i] = aux1 == aux2;
        acc += aux1;
        dataCovariance[i] =  Data(dataCovariance_input[i]);
    }
    length_state = acc;
    stateCovariance = new(std::nothrow) double[length_state*length_state];
    for(unsigned i = 0u; i < length_state*length_state; i++){
        stateCovariance[i] = 0.0;
    }
    unsigned offset = 0u;
    for(unsigned i = 0u; i < length_data; i++){
        unsigned aux = dataCovariance_input[i].GetLength();
        if(compactForm[i]){
            for(unsigned j = 0u; j < aux; j++){
                stateCovariance[(j+offset)*length_state+(j+offset)] = dataCovariance_input[i][j];
            }
        } else {
            for(unsigned j = 0u; j < aux; j++){
                for(unsigned k = 0u; k < aux; j++){
                    stateCovariance[(j+offset)*length_state+(k+offset)] = dataCovariance_input[i][j*aux+k];
                }
            }
        }
        offset += aux;
    }
}

PointCovariance::~PointCovariance(){
    delete[] stateCovariance;
    delete[] compactForm;
    delete[] dataCovariance;
}

void PointCovariance::UpdateArrayFromData()
{
    for(unsigned i = 0u; i < length_state*length_state; i++){
        stateCovariance[i] = 0.0;
    }
    unsigned offset = 0u;
    for(unsigned i = 0u; i < length_data; i++){
        unsigned aux = dataCovariance[i].GetLength();
        if(compactForm[i]){
            for(unsigned j = 0u; j < aux; j++){
                stateCovariance[(j+offset)*length_state+(j+offset)] = dataCovariance[i][j];
            }
        } else {
            for(unsigned j = 0u; j < aux; j++){
                for(unsigned k = 0u; k < aux; j++){
                    stateCovariance[(j+offset)*length_state+(k+offset)] = dataCovariance[i][j*aux+k];
                }
            }
        }
    }
}

void PointCovariance::UpdateDataFromArray()
{
    unsigned offset = 0u;
    for(unsigned i = 0u; i < length_data; i++){
        unsigned aux = dataCovariance[i].GetLength();
        if(compactForm[i]){
            for(unsigned j = 0u; j < aux; j++){
                dataCovariance[i][j] = stateCovariance[(j+offset)*length_state+(j+offset)];
            }
        } else {
            for(unsigned j = 0u; j < aux; j++){
                for(unsigned k = 0u; k < aux; j++){
                    dataCovariance[i][j*aux+k] = stateCovariance[(j+offset)*length_state+(k+offset)];
                }
            }
        }
    }
}

unsigned PointCovariance::GetLengthState(){
    return length_state;
}

unsigned PointCovariance::GetLengthData(){
    return length_data;
}

Data* PointCovariance::GetDataCovariance(){
    return dataCovariance;
}

double* PointCovariance::GetStateCovariance(){
    return stateCovariance;
}