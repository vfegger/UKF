#ifndef HEAT_FLUX_ESTIMATION_HEADER
#define HEAT_FLUX_ESTIMATION_HEADER

#include "HeatFluxEstimationMemory.hpp"

class HeatFluxEstimation
{
private:
    Pointer<HeatFluxEstimationMemory> memory;
    Pointer<Parameter> parameter;
    Pointer<Data> input;
    Pointer<DataCovariance> inputCovariance;
    Pointer<DataCovariance> inputNoise;
    Pointer<Data> measure;
    Pointer<DataCovariance> measureNoise;

public:
    Pointer<HeatFluxEstimationMemory> GetMemory();

    HeatFluxEstimation(unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt, double Sx, double Sy, double Sz, double St, double T0, double sT0, double sTm0, double Q0, double sQ0, double Amp, PointerType type_in, PointerContext context_in);

    void UpdateMeasure(Pointer<double> T_in, unsigned Lx, unsigned Ly, PointerType type_in, PointerContext context_in);

    ~HeatFluxEstimation();
};

#endif