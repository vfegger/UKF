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
        acc += (compactForm[i]) ? aux2 * aux2: aux2;
        dataCovariance[i] =  Data(dataCovariance_input[i]);
    }
    length_state = acc;
    stateCovariance = new(std::nothrow) double[length_state];
    for(unsigned i = 0u; i < length_state; i++){
        stateCovariance[i] = 0.0;
    }
    for(unsigned i = 0u; i < length_data; i++){
        unsigned aux = dataCovariance_input[i].GetLength();
        if(compactForm[i]){
            for(unsigned j = 0u; j < aux; j++){
                stateCovariance[j*length_state+j] = dataCovariance_input[i][j];
            }
        } else {
            for(unsigned j = 0u; j < aux; j++){
                for(unsigned k = 0u; k < aux; j++){
                    stateCovariance[j*length_state+k] = dataCovariance_input[i][j*aux+k];
                }
            }
        }
    }
}

PointCovariance::~PointCovariance(){
    delete[] stateCovariance;
    delete[] compactForm;
    delete[] dataCovariance;
}

void PointCovariance::UpdateArrayFromData()
{
    for(unsigned i = 0u; i < length_state; i++){
        stateCovariance[i] = 0.0;
    }
    for(unsigned i = 0u; i < length_data; i++){
        unsigned aux = dataCovariance[i].GetLength();
        if(compactForm[i]){
            for(unsigned j = 0u; j < aux; j++){
                stateCovariance[j*length_state+j] = dataCovariance[i][j];
            }
        } else {
            for(unsigned j = 0u; j < aux; j++){
                for(unsigned k = 0u; k < aux; j++){
                    stateCovariance[j*length_state+k] = dataCovariance[i][j*aux+k];
                }
            }
        }
    }
}

void PointCovariance::UpdateDataFromArray()
{
    for(unsigned i = 0u; i < length_data; i++){
        unsigned aux = dataCovariance[i].GetLength();
        if(compactForm[i]){
            for(unsigned j = 0u; j < aux; j++){
                dataCovariance[i][j] = stateCovariance[j*length_state+j];
            }
        } else {
            for(unsigned j = 0u; j < aux; j++){
                for(unsigned k = 0u; k < aux; j++){
                    dataCovariance[i][j*aux+k] = stateCovariance[j*length_state+k];
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