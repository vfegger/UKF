#ifndef HEAT_FLUX_GENERATOR_HEADER
#define HEAT_FLUX_GENERATOR_HEADER

#include <iostream>
#include <random>
#include "../../structure/include/MemoryHandler.hpp"

class HeatFluxGenerator
{
private:
    unsigned Lx, Ly, Lz, Lt;
    double Sx, Sy, Sz, St;
    double dx, dy, dz, dt;
    Pointer<double> Q;
    Pointer<double> T;
    double Amp;
    
    double K(double T_in);
    double C(double T_in);
    double DifferentialK(double TN_in, double T_in, double TP_in, double delta_in);
    void Evolution(double* T_in, double* T_out);
public:
    HeatFluxGenerator(unsigned Lx_in, unsigned Ly_in, unsigned Lz_in, unsigned Lt_in, double Sx_in, double Sy_in, double Sz_in, double St_in, double T0, double Amp_in, PointerType type_in, PointerContext context_in);
    void Generate(double mean_in, double sigma_in);
    void AddError(double mean_in, double sigma_in);
    void SetFlux(double t_in);
    Pointer<double> GetTemperature(unsigned t_in);
    ~HeatFluxGenerator();
};

#endif