#include "../include/HeatFluxGenerator.hpp"

HeatFluxGenerator::HeatFluxGenerator(unsigned Lx_in, unsigned Ly_in, unsigned Lz_in, unsigned Lt_in, double Sx_in, double Sy_in, double Sz_in, double St_in, double T0, double Amp_in)
{
    PointerType type_in = PointerType::CPU;
    PointerContext context_in = PointerContext::GPU_Aware;

    Lx = Lx_in;
    Ly = Ly_in;
    Lz = Lz_in;
    Lt = Lt_in;
    Sx = Sx_in;
    Sy = Sy_in;
    Sz = Sz_in;
    St = St_in;
    dx = Sx_in / Lx_in;
    dy = Sy_in / Ly_in;
    dz = Sz_in / Lz_in;
    dt = St_in / Lt_in;
    T = MemoryHandler::Alloc<double>(Lx * Ly * Lz * (Lt + 1), type_in, context_in);
    MemoryHandler::Set<double>(T, T0, 0, Lx * Ly * Lz);
    Q = MemoryHandler::Alloc<double>(Lx * Ly, type_in, context_in);
    Amp = Amp_in;
}

inline double HeatFluxGenerator::C(double T)
{
    return 1324.75 * T + 3557900.0;
}
inline double HeatFluxGenerator::K(double T)
{
    return 12.45 + (14e-3 + 2.517e-6 * T) * T;
}
double HeatFluxGenerator::DifferentialK(double TN_in, double T_in, double TP_in, double delta_in)
{
    double auxN = 2.0 * (K(TN_in) * K(T_in)) / (K(TN_in) + K(T_in)) * (TN_in - T_in) / delta_in;
    double auxP = 2.0 * (K(TP_in) * K(T_in)) / (K(TP_in) + K(T_in)) * (TP_in - T_in) / delta_in;
    return (auxN + auxP) / delta_in;
}

void HeatFluxGenerator::Evolution(double *T_in, double *T_out)
{
    unsigned index;
    double acc;
    double TiN, TiP;
    double TjN, TjP;
    double TkN, TkP;
    double T0;
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Ly; j++)
        {
            for (unsigned i = 0u; i < Lx; i++)
            {
                index = (k * Ly + j) * Lx + i;
                T0 = T_in[index];
                TiN = (i != 0u) ? T_in[index - 1] : T0;
                TiP = (i != Lx - 1) ? T_in[index + 1] : T0;
                TjN = (j != 0u) ? T_in[index - Lx] : T0;
                TjP = (j != Ly - 1) ? T_in[index + Lx] : T0;
                TkN = (k != 0u) ? T_in[index - Ly * Lx] : T0;
                TkP = (k != Lz - 1) ? T_in[index + Ly * Lx] : T0;
                acc = 0.0;
                // X dependency
                acc += DifferentialK(TiN, T0, TiP, dx);
                // Y dependency
                acc += DifferentialK(TjN, T0, TjP, dy);
                // Z dependency
                acc += DifferentialK(TkN, T0, TkP, dz);
                if (k == Lz - 1)
                {
                    acc += Amp * Q.pointer[j * Lx + i] / dz;
                }
                T_out[index] = T0 + dt * acc / C(T0);
            }
        }
    }
}

void HeatFluxGenerator::SetFlux(double t_in)
{
    unsigned index;
    for (unsigned j = 0u; j < Ly; j++)
    {
        for (unsigned i = 0u; i < Lx; i++)
        {
            index = j * Lx + i;
            if ((i + 0.5) * dx >= 0.4 * Sx && (i + 0.5) * dx <= 0.7 * Sx &&
                (j + 0.5) * dy >= 0.4 * Sy && (j + 0.5) * dy <= 0.7 * Sy)
            {
                Q.pointer[index] = 100.0;
            }
            else
            {
                Q.pointer[index] = 0.0;
            }
        }
    }
}

void HeatFluxGenerator::Generate(double mean_in, double sigma_in)
{
    for (unsigned t = 0u; t < Lt; t++)
    {
        SetFlux(t * dt);
        Evolution(T.pointer + t * Lx * Ly * Lz, T.pointer + (t + 1) * Lx * Ly * Lz);
    }
    AddError(mean_in, sigma_in);
}

void HeatFluxGenerator::AddError(double mean_in, double sigma_in)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean_in, sigma_in);
    for (unsigned i = 0u; i < Lx * Ly * Lz * Lt; i++)
    {
        T.pointer[i] += distribution(generator);
    }
}

Pointer<double> HeatFluxGenerator::GetTemperature(unsigned t_in)
{
    if (t_in > Lt)
    {
        std::cout << "Error: Out of range.\n";
        return Pointer<double>();
    }
    return Pointer<double>(T.pointer + t_in * Lx * Ly * Lz, T.type, T.context);
}

HeatFluxGenerator::~HeatFluxGenerator()
{
    MemoryHandler::Free<double>(T);
    MemoryHandler::Free<double>(Q);
}