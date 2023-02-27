#include "../include/HeatFluxEstimationMemory.hpp"

HFE_CRCMemory::HFE_CRCMemory() : UKFMemory()
{
}

HFE_CRCMemory::HFE_CRCMemory(Data &a, DataCovariance &b, DataCovariance &c, Data &d, DataCovariance &e, Parameter &f, PointerType type_in, PointerContext context_in) : UKFMemory(a, b, c, d, e, f, type_in, context_in)
{
}

HFE_CRCMemory::HFE_CRCMemory(const HFE_CRCMemory &memory_in) : UKFMemory(memory_in)
{
}

void HFE_CRCMemory::Evolution(Data &data_inout, Parameter &parameter_in, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in)
{
    HCRC::HCRCProblem problem;
    Pointer<unsigned> length = parameter_in.GetPointer<unsigned>(0u);
    unsigned Lr = length.pointer[0u];
    unsigned Lth = length.pointer[1u];
    unsigned Lz = length.pointer[2u];
    unsigned Lt = length.pointer[3u];
    problem.Lr = Lr;
    problem.Lth = Lth;
    problem.Lz = Lz;
    problem.Lt = Lt;
    Pointer<double> delta = parameter_in.GetPointer<double>(1u);
    problem.dr = delta.pointer[0u];
    problem.dth = delta.pointer[1u];
    problem.dz = delta.pointer[2u];
    problem.dt = delta.pointer[3u];
    Pointer<double> parms = parameter_in.GetPointer<double>(3u);
    problem.amp = parms.pointer[0u];
    problem.r0 = parms.pointer[1u];
    Pointer<double> pointer = data_inout.GetPointer();
    Pointer<double> T_inout = data_inout[0u];
    Pointer<double> Q_in = data_inout[1u];
    double *workspace = NULL;
    if (pointer.type == PointerType::CPU)
    {
        HCRC::CPU::AllocWorkspaceEuler(workspace, problem.Lr * problem.Lth * problem.Lz);
        HCRC::CPU::Euler(T_inout.pointer, T_inout.pointer, Q_in.pointer, problem, workspace);
        HCRC::CPU::FreeWorkspaceEuler(workspace);
    }
    else if (pointer.type == PointerType::GPU)
    {
        HCRC::GPU::AllocWorkspaceRK4(workspace, problem.Lr * problem.Lth * problem.Lz, stream_in);
        HCRC::GPU::RK4(T_inout.pointer, T_inout.pointer, Q_in.pointer, problem, workspace, cublasHandle_in, stream_in);
        HCRC::GPU::FreeWorkspaceRK4(workspace, stream_in);
    }
}

void HFE_CRCMemory::Observation(Data &data_in, Parameter &parameter_in, Data &data_out, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in)
{
    Pointer<unsigned> length = parameter_in.GetPointer<unsigned>(0u);
    MemoryHandler::Copy(data_out[0u], data_in[0u], length.pointer[1u] * length.pointer[2u]);
}