#ifndef HEAT_FLUX_ESTIMATION_MEMORY_HEADER
#define HEAT_FLUX_ESTIMATION_MEMORY_HEADER

#include "../../ukf/include/UKFMemory.hpp"

class HeatFluxEstimationMemory : public UKFMemory
{
public:
    HeatFluxEstimationMemory();
    HeatFluxEstimationMemory(Data &a, DataCovariance &b, DataCovariance &c, Data &d, DataCovariance &e, Parameter &f, PointerType type_in, PointerContext context_in);
    HeatFluxEstimationMemory(const HeatFluxEstimationMemory &memory_in);

    inline double C(double T);
    inline double K(double T);

    double DifferentialK(double TN, double T, double TP, double delta);

    void Evolution(Data &data_inout, Parameter &parameter_in) override;
    void Observation(Data &data_in, Parameter &parameter_in, Data &data_out) override;
};

#endif