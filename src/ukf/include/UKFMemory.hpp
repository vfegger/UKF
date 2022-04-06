#ifndef INPUT_HEADER
#define INPUT_HEADER

#include "../../structure/include/Data.hpp"
#include "../../structure/include/Parameter.hpp"

class UKFMemory
{
private:
    Parameter* parameter;

    Data* state;
    Data* stateCovariance;
    Data* stateNoise;

    Data* measureData;
    Data* measureDataNoise;
protected:

public:
    UKFMemory(Data& inputData_in, Data& inputDataCovariance_in, Data& inputDataNoise_in, Data& measureData_in, Data& measureDataNoise_in, Parameter& inputParameter_in);
    virtual ~UKFMemory();

    Parameter* GetParameter();

    Data* GetState();
    Data* GetStateCovariance();
    Data* GetStateNoise();

    Data* GetMeasure();
    void UpdateMeasure(Data& measureData_in);
    Data* GetMeasureNoise();

    virtual void Evolution(Data& data_inout, Parameter& parameter_in) = 0;
    virtual void Observation(Data& data_in, Parameter& parameter_in, Data& data_out) = 0;

};


#endif