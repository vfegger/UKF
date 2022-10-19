#ifndef HEAT_FLUX_ESTIMATION_HEADER
#define HEAT_FLUX_ESTIMATION_HEADER

#include "HeatFluxEstimationMemory.hpp"

class HeatFluxEstimation {
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

    HeatFluxEstimation(unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt, double Sx, double Sy, double Sz, double St);

    void UpdateMeasure(Pointer<double> T_in, unsigned Lx, unsigned Ly);

    ~HeatFluxEstimation();
};


#endif