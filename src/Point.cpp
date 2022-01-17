#include "Point.hpp"
#include <new>
#include <iostream>

Point::Point(){
    data = NULL;
    state = NULL;
    length_data = 0u;
    length_state = 0u;
}

Point::Point(Data* data_input, unsigned length_input){
    length_data = length_input;
    data = new(std::nothrow) Data[length_data];
    unsigned acc = 0u;
    for(unsigned i = 0u; i < length_data; i++){
        acc += data_input[i].GetLength();
        data[i] = Data(data_input[i]);
    }
    length_state = acc;
    state = new(std::nothrow) double[length_state];
    unsigned offset = 0u;
    for(unsigned i = 0u; i < length_data; i++){
        unsigned auxLength = data[i].GetLength();
        for(unsigned k = 0; k < auxLength; k++){
            state[k+offset] = data[i][k];
        }
        offset += auxLength;
    }
}

Point::Point(Point* point){
    length_data = point->length_data;
    length_state = point->length_state;
    data = new(std::nothrow) Data[length_data];
    for(unsigned i = 0u; i < length_data; i++){
        data[i] =  Data(point->data[i]);
    }
    state = new(std::nothrow) double[length_state];
    for(unsigned i = 0u; i < length_state; i++){
        state[i] = point->state[i];
    }
}

Point::Point(Point& point){
    length_data = point.length_data;
    length_state = point.length_state;
    data = new(std::nothrow) Data[length_data];
    for(unsigned i = 0u; i < length_data; i++){
        data[i] = Data(point.data[i]);
    }
    state = new(std::nothrow) double[length_state];
    for(unsigned i = 0u; i < length_data; i++){
        state[i] = point.state[i];
    }
}

Point::~Point(){
    delete[] state;
    delete[] data;
}

Point& Point::operator=(const Point& point_input){
    length_data = point_input.length_data;
    length_state = point_input.length_state;
    if(length_data == 0u){
        data = NULL;
        std::cout << "Null pointer\n";
        return *this;
    }
    if(length_state == 0u){
        state = NULL;
        std::cout << "Null pointer\n";
        return *this;
    }
    data = new(std::nothrow) Data[length_data];
    if(data == NULL){
        std::cout << "Vector of Data:\n";
        std::cout << "\tError in allocation of memory of size :" << length_data*sizeof(Data) << "\n";
        return *this;
    }
    state = new(std::nothrow) double[length_state];
    if(state == NULL){
        std::cout << "Vector of State:\n";
        std::cout << "\tError in allocation of memory of size :" << length_state*sizeof(double) << "\n";
        return *this;
    }
    for(unsigned i = 0u; i < length_data; i++){
        data[i] = Data(point_input.data[i]);
    }
    for(unsigned i = 0u; i < length_state; i++){
        state[i] = point_input.state[i];
    }
    return *this;
}

void Point::UpdateArrayFromData()
{
    unsigned offset = 0u;
    for(unsigned i = 0u; i < length_data; i++){
        unsigned auxLength = data[i].GetLength();
        for(unsigned k = 0; k < auxLength; k++){
            state[k+offset] = data[i][k];
        }
        offset += auxLength;
    }
}

void Point::UpdateDataFromArray()
{
    unsigned offset = 0u;
    for(unsigned i = 0; i < length_data; i++){
        unsigned auxLength = data[i].GetLength();
        for(unsigned k = 0; k < auxLength; k++){
            data[i][k] = state[k+offset];
        }
        offset += auxLength;
    }
}

unsigned Point::GetLengthState(){
    return length_state;
}

unsigned Point::GetLengthData(){
    return length_data;
}

Data* Point::GetData(){
    return data;
}

double* Point::GetState(){
    return state;
}
