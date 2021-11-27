#ifndef INPUT_HEADER
#define INPUT_HEADER

#include "Data.hpp"
#include "Parameters.hpp"
#include "State.hpp"

class Input
{
private:
    unsigned inputParametersLength;
    unsigned inputDataLength;
    Parameters* inputParameters;
    Data* inputData;
    Data* inputDataCovariance;
public:
    Input();
    virtual ~Input();
    void Initialize(Data* inputData_input, Data* inputDataCovariance_input, unsigned inputDataLength_input, Parameters* inputParameters_input, unsigned inputParametersLength_input);
    virtual void GetState(State* state_output);
    virtual State* GetState();
    virtual void GetCovariance(double* &covariance_output);
    virtual double* GetCovariance();
    virtual void GetParameters(Parameters* &parameters_output);
    virtual Parameters* GetParameters();
    virtual void Evolution(Data* inputData_input, Parameters* inputParameters_input) = 0;
    virtual void Observation(Data* inputData_input, Parameters* inputParameters_input, Data* observationData_output) = 0;
};

#endif