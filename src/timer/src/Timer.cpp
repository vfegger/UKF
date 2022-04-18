#include "../include/Timer.hpp"

Timer::Timer(unsigned length_in) : clock(){
    startTime = Time::now();
    length = length_in;
    if(length_in == 0u){
        std::cout << "Timer: Invalid lenght.\n";
    }
    duration = new(std::nothrow) double[length_in];
    count = 0u;
}

void Timer::Start(){
    startTime = Time::now();
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
}
void Timer::Print(){
    for(unsigned i = 0u; i < count; i++){
        if(i == 0){
            std::cout << "Time duration " << i << ": " << duration[i]/1e+6 << " ms\n"; 
        } else {
            std::cout << "Time duration " << i << ": " << (duration[i] - duration[i-1u])/1e+6 << " ms\n"; 
        }
    }
}
Timer::~Timer(){
    delete[] duration;
}