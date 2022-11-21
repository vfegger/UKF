#include "../include/Timer.hpp"

Timer::Timer() : clock(){
    startTime = Time::now();
    length = 0u;
    duration = NULL;
    count = 0u;
    isValid = false;
}

Timer::Timer(unsigned length_in) : clock(){
    startTime = Time::now();
    length = length_in;
    if(length_in == 0u){
        std::cout << "Timer: Invalid lenght.\n";
    }
    duration = new(std::nothrow) double[length_in];
    count = 0u;
    isValid = false;
}

void Timer::Start(){
    startTime = Time::now();
    isValid = false;
}
void Timer::Save(){
    if(count >= length){
        std::cout << "Error: Insufficient size\n";
    }
    duration[count] = (Time::now()-startTime).count();
    count++;
}
void Timer::Reset(){
    startTime = Time::now();
    count = 0u;
    isValid = false;
}
void Timer::Print(){
    if(!isValid){
        SetValues();
    }
    for(unsigned i = 0u; i < count; i++){
        std::cout << "Time duration " << i << ": " << duration[i] << " ms\n"; 
    }
}
void Timer::SetValues(){
    if(isValid){
        return;
    }
    for(unsigned i = count-1u; i > 0u; i--){
        duration[i] -= duration[i-1u];
    }
    for(unsigned i = 0u; i < count; i++){
        duration[i] /= 1e+6;
    }
    isValid = true;
}
double* Timer::GetValues(){
    return duration;
}
unsigned Timer::GetCount(){
    return count;
}
Timer::~Timer(){
    delete[] duration;
}