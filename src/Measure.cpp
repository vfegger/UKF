#include "Measure.hpp"
#include <new>
#include <iostream>

Measure::Measure(Data* data_input, Data* dataCovariance_input, unsigned length_input){
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

Measure::~Measure(){
    delete[] pointCovariance;
    delete[] point;
}

unsigned Measure::GetStateLength(){
    return point->GetLengthState();
}

Point* Measure::GetPoint(){
    return point;
}

PointCovariance* Measure::GetPointCovariance(){
    return pointCovariance;
}