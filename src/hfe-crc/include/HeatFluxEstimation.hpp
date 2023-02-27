#ifndef HEAT_FLUX_ESTIMATION_HEADER
#define HEAT_FLUX_ESTIMATION_HEADER

#include "HeatFluxEstimationMemory.hpp"

class HFE_CRC
{
private:
    Pointer<HFE_CRCMemory> memory;
    Pointer<Parameter> parameter;
    Pointer<Data> input;
    Pointer<DataCovariance> inputCovariance;
    Pointer<DataCovariance> inputNoise;
    Pointer<Data> measure;
    Pointer<DataCovariance> measureNoise;

public:
    Pointer<HFE_CRCMemory> GetMemory();

    HFE_CRC(unsigned Lr, unsigned Lth, unsigned Lz, unsigned Lt, double Sr, double Sth, double Sz, double St, double T0, double sT0, double sTm0, double Q0, double sQ0, PointerType type_in, PointerContext context_in);

    void UpdateMeasure(Pointer<double> T_in, PointerType type_in, PointerContext context_in);

    ~HFE_CRC();
};

#endif