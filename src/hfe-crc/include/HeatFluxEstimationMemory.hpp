#ifndef HEAT_FLUX_ESTIMATION_MEMORY_HEADER
#define HEAT_FLUX_ESTIMATION_MEMORY_HEADER

#include "../../ukf/include/UKFMemory.hpp"
#include "HeatConductionRadiationCylinder.hpp"

class HFE_CRCMemory : public UKFMemory
{
public:
    HFE_CRCMemory();
    HFE_CRCMemory(Data &a, DataCovariance &b, DataCovariance &c, Data &d, DataCovariance &e, Parameter &f, PointerType type_in, PointerContext context_in);
    HFE_CRCMemory(const HFE_CRCMemory &memory_in);
    
    void Evolution(Data &data_inout, Parameter &parameter_in, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in = cudaStreamDefault) override;
    void Observation(Data &data_in, Parameter &parameter_in, Data &data_out, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in = cudaStreamDefault) override;
};

#endif