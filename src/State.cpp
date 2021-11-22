#include "State.hpp"
#include <new>
#include <iostream>

State::State(Data* data_input, Data* dataCovariance_input, unsigned length_input){
    length_data = length_input;
    point = new(std::nothrow) Point(data_input,length_data);
    if(point == NULL){
        std::cout << "State is NULL with length = " << length_input << "\n";
    }
    pointCovariance = new(std::nothrow) PointCovariance(data_input,dataCovariance_input,length_data);
    if(pointCovariance == NULL){
        std::cout << "State Covariance is NULL with length = " << length_input << "\n";
    }
}

State::~State(){
    delete[] pointCovariance;
    delete[] point;
}

unsigned State::GetStateLength(){
    return point->GetLengthState();
}

Point* State::GetPoint(){
    return point;
}

PointCovariance* State::GetPointCovariance(){
    return pointCovariance;
}