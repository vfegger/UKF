#ifndef UKF_HEADER
#define UKF_HEADER

#include "Input.hpp"
#include "Output.hpp"

class UKF
{
private:
    Input* input;
    void EvolutionFunction(InputParameters* parms, InputData* data);
    void ObservationFunction(InputParameters* parms, InputData* data);
    Output* output;
public:
    UKF();
    ~UKF();
    void Initialize(Input* input);
    void Solve();
    void Export(Output* output);
};
#endif