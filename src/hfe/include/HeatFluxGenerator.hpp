#ifndef HEAT_FLUX_GENERATOR_HEADER
#define HEAT_FLUX_GENERATOR_HEADER

#include <iostream>
#include <random>
#include "../../structure/include/MemoryHandler.hpp"
#include "HeatConduction.hpp"

class HeatFluxGenerator
{
private:
    HeatConduction::HeatConductionProblem problem;

    Pointer<double> Q;
    Pointer<double> T;

public:
    HeatFluxGenerator(unsigned Lx_in, unsigned Ly_in, unsigned Lz_in, unsigned Lt_in, double Sx_in, double Sy_in, double Sz_in, double St_in, double T0_in, double Q0_in, double Amp_in, PointerType type_in, PointerContext context_in);
    void Generate(double mean_in, double sigma_in);
    void AddError(double mean_in, double sigma_in);
    Pointer<double> GetTemperature(unsigned t_in);
    ~HeatFluxGenerator();
};

#endif