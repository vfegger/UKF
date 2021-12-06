#include "Measure.hpp"
#include <new>
#include <iostream>

Measure::Measure(Data* dataReal_input, Data* data_input, Data* dataCovariance_input, unsigned length_input){
    length_data = length_input;
    realPoint = new(std::nothrow) Point(dataReal_input,length_data);
    if(realPoint == NULL){
        std::cout << "Real measure is NULL with length = " << length_input << "\n";
    }
    pointNoise = new(std::nothrow) PointCovariance(data_input,dataCovariance_input,length_data);
    if(pointNoise == NULL){
        std::cout << "Measure Noise is NULL with length = " << length_input << "\n";        
    }
    point = new(std::nothrow) Point(data_input,length_data);
    if(point == NULL){
        std::cout << "Measure is NULL with length = " << length_input << "\n";
    }
    pointCovariance = new(std::nothrow) PointCovariance(data_input,dataCovariance_input,length_data);
    if(pointCovariance == NULL){
        std::cout << "Measure Covariance is NULL with length = " << length_input << "\n";
    }
}

Measure::~Measure(){
    delete[] pointCovariance;
    delete[] point;
    delete[] pointNoise;
    delete[] realPoint;
}

unsigned Measure::GetStateLength(){
    return point->GetLengthState();
}

Point* Measure::GetPoint(){
    return point;
}

Point* Measure::GetRealPoint(){
    return realPoint;
}

PointCovariance* Measure::GetPointCovariance(){
    return pointCovariance;
}