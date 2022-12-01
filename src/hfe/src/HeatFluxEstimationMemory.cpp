#include "../include/HeatFluxEstimationMemory.hpp"

HeatFluxEstimationMemory::HeatFluxEstimationMemory() : UKFMemory()
{
}

HeatFluxEstimationMemory::HeatFluxEstimationMemory(Data &a, DataCovariance &b, DataCovariance &c, Data &d, DataCovariance &e, Parameter &f, PointerType type_in, PointerContext context_in) : UKFMemory(a, b, c, d, e, f, type_in, context_in)
{
}

HeatFluxEstimationMemory::HeatFluxEstimationMemory(const HeatFluxEstimationMemory &memory_in) : UKFMemory(memory_in)
{
}

void HeatFluxEstimationMemory::Evolution(Data &data_inout, Parameter &parameter_in, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in)
{
    HeatConduction::HeatConductionProblem problem;
    Pointer<unsigned> length = parameter_in.GetPointer<unsigned>(0u);
    unsigned Lx = length.pointer[0u];
    unsigned Ly = length.pointer[1u];
    unsigned Lz = length.pointer[2u];
    unsigned Lt = length.pointer[3u];
    problem.Lx = Lx;
    problem.Ly = Ly;
    problem.Lz = Lz;
    problem.Lt = Lt;
    Pointer<double> delta = parameter_in.GetPointer<double>(1u);
    problem.dx = delta.pointer[0u];
    problem.dy = delta.pointer[1u];
    problem.dz = delta.pointer[2u];
    problem.dt = delta.pointer[3u];
    problem.amp = parameter_in.GetPointer<double>(3u).pointer[0u];
    Pointer<double> pointer = data_inout.GetPointer();
    Pointer<double> T_inout = data_inout[0u];
    Pointer<double> Q_in = data_inout[1u];
    double *workspace = NULL;
    if (pointer.type == PointerType::CPU)
    {
        HeatConduction::CPU::AllocWorkspaceEuler(workspace, problem.Lx * problem.Ly * problem.Lz);
        HeatConduction::CPU::Euler(T_inout.pointer, T_inout.pointer, Q_in.pointer, problem, workspace);
        HeatConduction::CPU::FreeWorkspaceEuler(workspace);
    }
    else if (pointer.type == PointerType::GPU)
    {
        HeatConduction::GPU::AllocWorkspaceEuler(workspace, problem.Lx * problem.Ly * problem.Lz, stream_in);
        HeatConduction::GPU::Euler(T_inout.pointer, T_inout.pointer, Q_in.pointer, problem, workspace, cublasHandle_in, stream_in);
        HeatConduction::GPU::FreeWorkspaceEuler(workspace, stream_in);
    }
}

void HeatFluxEstimationMemory::Observation(Data &data_in, Parameter &parameter_in, Data &data_out, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in)
{
    Pointer<unsigned> length = parameter_in.GetPointer<unsigned>(0u);
    MemoryHandler::Copy(data_out[0u], data_in[0u], length.pointer[0u] * length.pointer[1u]);
}