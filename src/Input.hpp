#ifndef INPUT_HEADER
#define INPUT_HEADER

#include "Data.hpp"
#include "Parameters.hpp"
#include "State.hpp"
#include "Measure.hpp"

class Input
{
private:
    unsigned inputDataLength;
    unsigned measureDataLength;
    Parameters* inputParameters;
    Data* inputData;
    Data* inputDataCovariance;
    Data* inputDataNoise;
    Data* measureData;
    Data* measureDataNoise;
public:
    Input();
    virtual ~Input();
    void Initialize(Data* inputData_input, Data* inputDataCovariance_input, Data* inputDataNoise_input, unsigned inputDataLength_input, Parameters* inputParameters_input, Data* measureData_input, Data* measureDataNoise_input, unsigned measureDataLength_input);
    virtual void GetState(State* state_output);
    virtual State* GetState();
    virtual void GetCovariance(double* &covariance_output);
    virtual double* GetCovariance();
    virtual void GetParameters(Parameters* &parameters_output);
    virtual Parameters* GetParameters();
    virtual void GetMeasure(Measure* &measure_output);
    virtual Measure* GetMeasure();
    virtual void Evolution(Data* inputData_input, Parameters* inputParameters_input) = 0;
    virtual void Observation(Data* inputData_input, Parameters* inputParameters_input, Data* observationData_output) = 0;
};

#endif