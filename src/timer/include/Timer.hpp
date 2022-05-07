#ifndef TIMER_HEADER
#define TIMER_HEADER

#include <iostream>
#include <fstream>
#include <chrono>

using Time = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;
using TimePoint = std::chrono::time_point<Time,Duration>;

class Timer
{
private:
    Time clock;
    TimePoint startTime;
    double* duration;
    unsigned count;
    unsigned length;

public:
    Timer(unsigned length_in);

    void Start();
    void Save();
    void Reset();
    void Print();
    double* GetValues();
    unsigned GetCount();



    ~Timer();
};

#endif
