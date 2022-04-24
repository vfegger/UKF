#ifndef HEAT_FLUX_ESTIMATION_HEADER
#define HEAT_FLUX_ESTIMATION_HEADER

#include "HeatFluxEstimationMemory.hpp"

class HeatFluxEstimation {
private:
    HeatFluxEstimationMemory* memory;
    Parameter* parameter;
    Data* input;
    DataCovariance* inputCovariance;
    DataCovariance* inputNoise;
    Data* measure;
    DataCovariance* measureNoise;
public:
    HeatFluxEstimationMemory* GetMemory();

    HeatFluxEstimation(unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt, double Sx, double Sy, double Sz, double St);

    void UpdateMeasure(double* T_in, unsigned Lx, unsigned Ly);

    ~HeatFluxEstimation();
};


#endif