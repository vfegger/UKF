#include "Point.hpp"
#include <new>

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
}

Point::Point(Point* point){
    length_data = point->length_data;
    length_state = point->length_state;
    data = new(std::nothrow) Data[length_data];
    for(unsigned i = 0u; i < length_data; i++){
        data[i] =  Data(point->data[i]);
    }
    state = new(std::nothrow) double[length_state];
    for(unsigned i = 0u; i < length_data; i++){
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

void Point::UpdateArrayFromData()
{
    for(unsigned i = 0, j = 0u; i < length_data; i++){
        unsigned auxLength = data[i].GetLength();
        for(unsigned k = 0; k < auxLength; k++,j++){
            state[j] = data[i][k];
        }
    }
}

void Point::UpdateDataFromArray()
{
    for(unsigned i = 0, j = 0u; i < length_data; i++){
        unsigned auxLength = data[i].GetLength();
        for(unsigned k = 0; k < auxLength; k++,j++){
            data[i][k] = state[j];
        }
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
