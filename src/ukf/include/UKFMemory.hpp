#ifndef UKF_MEMORY_HEADER
#define UKF_MEMORY_HEADER

#include "../../structure/include/Data.hpp"
#include "../../structure/include/DataCovariance.hpp"
#include "../../structure/include/Parameter.hpp"

class UKFMemory
{
private:
    Parameter* parameter;

    Data* state;
    DataCovariance* stateCovariance;
    DataCovariance* stateNoise;

    Data* measureData;
    DataCovariance* measureDataNoise;
protected:

public:
    UKFMemory(Data& inputData_in, DataCovariance& inputDataCovariance_in, DataCovariance& inputDataNoise_in, Data& measureData_in, DataCovariance& measureDataNoise_in, Parameter& inputParameter_in);
    virtual ~UKFMemory();

    Parameter* GetParameter();

    Data* GetState();
    DataCovariance* GetStateCovariance();
    DataCovariance* GetStateNoise();

    Data* GetMeasure();
    void UpdateMeasure(Data& measureData_in);
    DataCovariance* GetMeasureNoise();

    virtual void Evolution(Data& data_inout, Parameter& parameter_in) = 0;
    virtual void Observation(Data& data_in, Parameter& parameter_in, Data& data_out) = 0;

    double* GetWeightMean();
    double* GetWeightCovariance();
};


#endif