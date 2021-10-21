#ifndef UKF_HEADER
#define UKF_HEADER

#include "Input.hpp"

class UKF
{
private:
    InputData* inputData;
    InputParameters* inputParameters;
    void EvolutionFunction(InputParameters* parms, InputData* data);
    void ObservationFunction(InputParameters* parms, InputData* data);
public:
    UKF();
    ~UKF();
};
#endif