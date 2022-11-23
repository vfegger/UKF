#ifndef UKF_MEMORY_HEADER
#define UKF_MEMORY_HEADER

#include "../../structure/include/Data.hpp"
#include "../../structure/include/DataCovariance.hpp"
#include "../../structure/include/Parameter.hpp"
#include "../../structure/include/MemoryHandler.hpp"

class UKFMemory
{
private:
    Pointer<Parameter> parameter;

    Pointer<Data> state;
    Pointer<DataCovariance> stateCovariance;
    Pointer<DataCovariance> stateNoise;

    Pointer<Data> measureData;
    Pointer<DataCovariance> measureDataNoise;

    PointerType type;
    PointerContext context;

protected:
public:
    UKFMemory();
    UKFMemory(Data &inputData_in, DataCovariance &inputDataCovariance_in, DataCovariance &inputDataNoise_in, Data &measureData_in, DataCovariance &measureDataNoise_in, Parameter &inputParameter_in, PointerType type_in, PointerContext context_in);
    UKFMemory(const UKFMemory& memory_in);
    UKFMemory& operator=(const UKFMemory& memory_in);
    virtual ~UKFMemory();

    Pointer<Parameter> GetParameter();

    Pointer<Data> GetState();
    Pointer<DataCovariance> GetStateCovariance();
    Pointer<DataCovariance> GetStateNoise();

    Pointer<Data> GetMeasure();
    void UpdateMeasure(Data &measureData_in);
    Pointer<DataCovariance> GetMeasureNoise();

    virtual void Evolution(Data &data_inout, Parameter &parameter_in) = 0;
    virtual void Observation(Data &data_in, Parameter &parameter_in, Data &data_out) = 0;

    Pointer<double> GetWeightMean();
    Pointer<double> GetWeightCovariance();
};

#endif