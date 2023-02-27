#ifndef HEAT_FLUX_GENERATOR_HEADER
#define HEAT_FLUX_GENERATOR_HEADER

#include <iostream>
#include <random>
#include "../../structure/include/MemoryHandler.hpp"
#include "HeatConductionRadiationCylinder.hpp"

class HFG_CRC
{
private:
    HCRC::HCRCProblem problem;

    Pointer<double> Q;
    Pointer<double> T;

public:
    HFG_CRC(unsigned Lr_in, unsigned Lth_in, unsigned Lz_in, unsigned Lt_in, double Sr_in, double Sth_in, double Sz_in, double St_in, double T0_in, double Q0_in, double Amp_in, double r0_in, PointerType type_in, PointerContext context_in);
    void Generate(double mean_in, double sigma_in, cublasHandle_t handle_in, cudaStream_t stream_in);
    void AddError(double mean_in, double sigma_in);
    Pointer<double> GetTemperature(unsigned t_in);
    ~HFG_CRC();
};

#endif