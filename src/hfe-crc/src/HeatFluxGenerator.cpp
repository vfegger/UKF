#include "../include/HeatFluxGenerator.hpp"

HFG_CRC::HFG_CRC(unsigned Lr_in, unsigned Lth_in, unsigned Lz_in, unsigned Lt_in, double Sr_in, double Sth_in, double Sz_in, double St_in, double T0_in, double Q0_in, double Tamb0_in, double Amp_in, double r0_in, double h_in, PointerType type_in, PointerContext context_in, unsigned iteration) : problem(T0_in, Q0_in, Amp_in, r0_in, h_in, Sr_in, Sth_in, Sz_in, St_in, Lr_in, Lth_in, Lz_in, Lt_in, iteration)
{
    T = MemoryHandler::Alloc<double>(Lr_in * Lth_in * Lz_in * (Lt_in + 1), type_in, context_in);
    MemoryHandler::Set<double>(T, T0_in, 0u, Lr_in * Lth_in * Lz_in);
    Q = MemoryHandler::Alloc<double>(Lth_in * Lz_in, type_in, context_in);
    Tamb = MemoryHandler::Alloc<double>(1u, type_in, context_in);
    MemoryHandler::Set<double>(Tamb, Tamb0_in, 0u, 1u);
}

void HFG_CRC::Generate(double mean_in, double sigma_in, cublasHandle_t handle_in, cudaStream_t stream_in)
{
    double *workspace = NULL;
    unsigned L = problem.Lr * problem.Lth * problem.Lz;
    if (T.type == PointerType::CPU)
    {
        HCRC::CPU::AllocWorkspaceRK4(workspace, L);
        for (unsigned t = 0u; t < problem.Lt; t++)
        {
            HCRC::CPU::SetFlux(Q.pointer, problem, t * problem.dt);
            HCRC::CPU::RK4(T.pointer + (t + 1) * L, T.pointer + t * L, Q.pointer, Tamb.pointer, problem, workspace);
        }
        HCRC::CPU::FreeWorkspaceRK4(workspace);
        HCRC::CPU::AddError(T.pointer, mean_in, sigma_in, (problem.Lt + 1) * L);
    }
    else if (T.type == PointerType::GPU)
    {
        HCRC::GPU::AllocWorkspaceRK4(workspace, L, stream_in);
        for (unsigned t = 0u; t < problem.Lt; t++)
        {
            HCRC::GPU::SetFlux(Q.pointer, problem, t * problem.dt, stream_in);
            HCRC::GPU::RK4(T.pointer + (t + 1) * L, T.pointer + t * L, Q.pointer, Tamb.pointer, problem, workspace, handle_in, stream_in);
        }
        HCRC::GPU::FreeWorkspaceRK4(workspace, stream_in);
        HCRC::GPU::AddError(T.pointer, mean_in, sigma_in, (problem.Lt + 1) * L, stream_in);
    }
}

Pointer<double> HFG_CRC::GetTemperature(unsigned t_in)
{
    if (t_in > problem.Lt)
    {
        std::cout << "Error: Out of range.\n";
        return Pointer<double>();
    }
    return Pointer<double>(T.pointer + t_in * problem.Lr * problem.Lth * problem.Lz, T.type, T.context);
}

void HFG_CRC::GetCompleteTemperatureBoundary(Pointer<double> &T_out, cudaStream_t stream_in)
{
    Pointer<double> in, out;
    for (unsigned t = 0u; t <= problem.Lt; t++)
    {
        for (unsigned k = 0u; k < problem.Lz; k++)
        {
            for (unsigned j = 0u; j < problem.Lth; j++)
            {
                in = Pointer<double>(T.pointer + HCRC::Index3D(0u, j, k, problem.Lr, problem.Lth, problem.Lz), T.type, T.context);
                out = Pointer<double>(T_out.pointer + ((t * problem.Lz + k) * problem.Lth + j), T_out.type, T_out.context);
                MemoryHandler::Copy(in, out, 1u, stream_in);
            }
        }
    }
}

HFG_CRC::~HFG_CRC()
{
    MemoryHandler::Free<double>(T);
    MemoryHandler::Free<double>(Q);
    MemoryHandler::Free<double>(Tamb);
}