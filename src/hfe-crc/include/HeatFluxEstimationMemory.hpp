#ifndef HEAT_FLUX_ESTIMATION_MEMORY_HEADER
#define HEAT_FLUX_ESTIMATION_MEMORY_HEADER

#include "../../ukf/include/UKFMemory.hpp"
#include "HeatConduction.hpp"

class HeatFluxEstimationMemory : public UKFMemory
{
public:
    HeatFluxEstimationMemory();
    HeatFluxEstimationMemory(Data &a, DataCovariance &b, DataCovariance &c, Data &d, DataCovariance &e, Parameter &f, PointerType type_in, PointerContext context_in);
    HeatFluxEstimationMemory(const HeatFluxEstimationMemory &memory_in);
    
    void Evolution(Data &data_inout, Parameter &parameter_in, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in = cudaStreamDefault) override;
    void Observation(Data &data_in, Parameter &parameter_in, Data &data_out, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in = cudaStreamDefault) override;
};

#endif