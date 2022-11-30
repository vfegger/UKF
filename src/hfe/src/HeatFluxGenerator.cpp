#include "../include/HeatFluxGenerator.hpp"

HeatFluxGenerator::HeatFluxGenerator(unsigned Lx_in, unsigned Ly_in, unsigned Lz_in, unsigned Lt_in, double Sx_in, double Sy_in, double Sz_in, double St_in, double T0_in, double Q0_in, double Amp_in, PointerType type_in, PointerContext context_in) : problem(T0_in, Q0_in, Amp_in, Sx_in, Sy_in, Sz_in, St_in, Lx_in, Ly_in, Lz_in, Lt_in)
{
    T = MemoryHandler::Alloc<double>(Lx_in * Ly_in * Lz_in * (Lt_in + 1), type_in, context_in);
    MemoryHandler::Set<double>(T, T0_in, 0, Lx_in * Ly_in * Lz_in);
    Q = MemoryHandler::Alloc<double>(Lx_in * Ly_in, type_in, context_in);
}

void HeatFluxGenerator::Generate(double mean_in, double sigma_in)
{
    double *workspace = NULL;
    unsigned L = problem.Lx * problem.Ly * problem.Lz;
    if (T.type == PointerType::CPU)
    {
        HeatConduction::CPU::AllocWorkspaceRK4(workspace, L);
        for (unsigned t = 0u; t < problem.Lt; t++)
        {
            HeatConduction::CPU::SetFlux(Q.pointer, problem, t * problem.dt);
            HeatConduction::CPU::RK4(T.pointer + (t + 1) * L, T.pointer + t * L, Q.pointer, problem, workspace);
            HeatConduction::CPU::FreeWorkspaceRK4(workspace);
        }
        HeatConduction::CPU::AddError(T.pointer, mean_in, sigma_in, (problem.Lt + 1) * L);
    }
    else if (T.type == PointerType::GPU)
    {
        HeatConduction::GPU::AllocWorkspaceRK4(workspace, L);
        for (unsigned t = 0u; t < problem.Lt; t++)
        {
            HeatConduction::GPU::SetFlux(Q.pointer, problem, t * problem.dt);
            HeatConduction::GPU::RK4(T.pointer + (t + 1) * L, T.pointer + t * L, Q.pointer, problem, workspace);
            HeatConduction::GPU::FreeWorkspaceRK4(workspace);
        }
        HeatConduction::GPU::AddError(T.pointer, mean_in, sigma_in, (problem.Lt + 1) * L);
    }
}

Pointer<double> HeatFluxGenerator::GetTemperature(unsigned t_in)
{
    if (t_in > problem.Lt)
    {
        std::cout << "Error: Out of range.\n";
        return Pointer<double>();
    }
    return Pointer<double>(T.pointer + t_in * problem.Lx * problem.Ly * problem.Lz, T.type, T.context);
}

HeatFluxGenerator::~HeatFluxGenerator()
{
    MemoryHandler::Free<double>(T);
    MemoryHandler::Free<double>(Q);
}