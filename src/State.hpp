#ifndef STATE_HEADER
#define STATE_HEADER

#include "Data.hpp"


class State
{
private:
    Data* data;
    double* state;
    unsigned length_data;
    unsigned length;
public:
    State(Data* data_input, unsigned length_input);
    ~State();
    void UpdateArrayFromData();
    void UpdateDataFromArray();
};

#endif