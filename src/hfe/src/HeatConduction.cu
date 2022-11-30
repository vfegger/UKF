#include "../include/HeatConduction.hpp"

HeatConduction::HeatConductionProblem::HeatConductionProblem() : T0(0.0), Q0(0.0), amp(0.0), Sx(0.0), Sy(0.0), Sz(0.0), St(0.0), Lx(0u), Ly(0u), Lz(0u), Lt(0u), dx(0.0), dy(0.0), dz(0.0), dt(0.0)
{
}

HeatConduction::HeatConductionProblem::HeatConductionProblem(double T0_in, double Q0_in, double amp_in, double Sx_in, double Sy_in, double Sz_in, double St_in, unsigned Lx_in, unsigned Ly_in, unsigned Lz_in, unsigned Lt_in) : T0(T0_in), Q0(Q0_in), amp(amp_in), Sx(Sx_in), Sy(Sy_in), Sz(Sz_in), St(St_in), Lx(Lx_in), Ly(Ly_in), Lz(Lz_in), Lt(Lt_in), dx(Sx_in / Lx_in), dy(Sy_in / Ly_in), dz(Sz_in / Lz_in), dt(St_in / Lt_in)
{
}

// CPU Section

inline double HeatConduction::CPU::C(double T_in)
{
    return (14e-3 + 2.517e-6 * T_in) * T_in + 12.45;
}

inline double HeatConduction::CPU::K(double T_in)
{
    return 1324.75 * T_in + 3557900.0;
}

double HeatConduction::CPU::DifferentialK(double TN_in, double T_in, double TP_in, double delta_in)
{
    double auxN = 2.0 * (K(TN_in) * K(T_in)) / (K(TN_in) + K(T_in)) * (TN_in - T_in) / delta_in;
    double auxP = 2.0 * (K(TP_in) * K(T_in)) / (K(TP_in) + K(T_in)) * (TP_in - T_in) / delta_in;
    return (auxN + auxP) / delta_in;
}

void HeatConduction::CPU::Differential(double *diff_out, const double *T_in, const double *Q_in, double amp, double dx, double dy, double dz, unsigned Lx, unsigned Ly, unsigned Lz)
{
    unsigned index, offset;
    double acc;
    double T0, TiN, TiP, TjN, TjP, TkN, TkP;
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

                diff_out[index] = acc;
            }
        }
    }
    offset = (Lz - 1) * Ly * Lx;
    for (unsigned j = 0u; j < Ly; j++)
    {
        for (unsigned i = 0u; i < Lx; i++)
        {
            index = j * Lx + i;
            diff_out[offset + index] = (amp / dz) * Q_in[index];
        }
    }
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Ly; j++)
        {
            for (unsigned i = 0u; i < Lx; i++)
            {
                index = (k * Ly + j) * Lx + i;
                diff_out[index] /= C(T_in[index]);
            }
        }
    }
}

void HeatConduction::CPU::AllocWorkspaceEuler(double *&workspace_out, unsigned length_in)
{
    workspace_out = new double[length_in];
}

void HeatConduction::CPU::FreeWorkspaceEuler(double *&workspace_in)
{
    delete[] workspace_in;
    workspace_in = NULL;
}

void HeatConduction::CPU::Euler(double *T_out, const double *T_in, double *Q_in, HeatConductionProblem &problem_in, double *workspace)
{
    unsigned L = problem_in.Lx * problem_in.Ly * problem_in.Lz;
    Differential(workspace, T_in, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);
    for (unsigned i = 0u; i < L; i++)
    {
        T_out[i] = T_in[i] + (problem_in.dt / 6.0) * (workspace[i]);
    }
}

void HeatConduction::CPU::AllocWorkspaceRK4(double *&workspace_out, unsigned length_in)
{
    workspace_out = new double[5u * length_in];
}

void HeatConduction::CPU::FreeWorkspaceRK4(double *&workspace_in)
{
    delete[] workspace_in;
    workspace_in = NULL;
}

void HeatConduction::CPU::RK4(double *T_out, double *T_in, double *Q_in, HeatConductionProblem &problem_in, double *workspace)
{
    unsigned L = problem_in.Lx * problem_in.Ly * problem_in.Lz;
    double *k1, *k2, *k3, *k4;
    double *T_aux;
    T_aux = workspace;
    k1 = T_aux + L;
    k2 = k1 + L;
    k3 = k2 + L;
    k4 = k3 + L;
    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i];
    }
    Differential(k1, T_aux, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i] + 0.5 * problem_in.dt * k1[i];
    }
    Differential(k2, T_aux, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);
    T_aux = T_in;

    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i] + 0.5 * problem_in.dt * k2[i];
    }
    Differential(k3, T_aux, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);
    T_aux = T_in;

    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i] + problem_in.dt * k3[i];
    }
    Differential(k4, T_aux, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_out[i] = T_in[i] + (problem_in.dt / 6.0) * (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]);
    }
}

void HeatConduction::CPU::SetFlux(double *Q_out, HeatConductionProblem &problem_in, unsigned t_in)
{
    bool xCondition, yCondition;
    double dx, dy, Sx, Sy;
    unsigned Lx, Ly;
    dx = problem_in.dx;
    dy = problem_in.dy;
    Sx = problem_in.Sx;
    Sy = problem_in.Sy;
    Lx = problem_in.Lx;
    Ly = problem_in.Ly;
    for (unsigned j = 0u; j < problem_in.Ly; j++)
    {
        for (unsigned i = 0u; i < problem_in.Lx; i++)
        {
            xCondition = (i + 0.5) * dx >= 0.4 * Sx && (i + 0.5) * dx <= 0.7 * Sx;
            yCondition = (j + 0.5) * dy >= 0.4 * Sy && (j + 0.5) * dy <= 0.7 * Sy;
            Q_out[j * Lx + i] = (xCondition && yCondition) ? 100.0 : 0.0;
        }
    }
}

void HeatConduction::CPU::AddError(double *T_out, double mean_in, double sigma_in, unsigned length)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean_in, sigma_in);
    for (unsigned i = 0u; i < length; i++)
    {
        T_out[i] += distribution(generator);
    }
}

// GPU Section

__device__ inline double C(double T_in)
{
    return (14e-3 + 2.517e-6 * T_in) * T_in + 12.45;
}

__device__ inline double K(double T_in)
{
    return 1324.75 * T_in + 3557900.0;
}

__device__ inline double DifferentialK(double TN_in, double T_in, double TP_in, double delta_in)
{
    double auxN = 2.0 * (K(TN_in) * K(T_in)) / (K(TN_in) + K(T_in)) * (TN_in - T_in) / delta_in;
    double auxP = 2.0 * (K(TP_in) * K(T_in)) / (K(TP_in) + K(T_in)) * (TP_in - T_in) / delta_in;
    return (auxN + auxP) / delta_in;
}

__device__ inline unsigned GetIndex(unsigned i, unsigned j, unsigned k, unsigned Li, unsigned Lj)
{
    return (k * Lj + j) * Li + i;
}

__global__ void DifferentialAxis(double *diff_out, const double *T_in, double dx, double dy, double dz, unsigned Lx, unsigned Ly, unsigned Lz)
{
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned index = GetIndex(xIndex, yIndex, zIndex, Lx, Ly);
    extern __shared__ double T[];
    unsigned thread;
    unsigned threadDimX;
    unsigned threadDimY;
    double T_aux = 0.0;
    double diff_aux = 0.0;
    if (xIndex < Lx && yIndex < Ly && zIndex < Lz)
    {
        T_aux = T_in[index];
    }
    thread = GetIndex(threadIdx.x + 1u, threadIdx.y + 1u, threadIdx.z + 1u, blockDim.x + 2u, blockDim.y + 2u);
    threadDimX = blockDim.x + 2u;
    threadDimY = blockDim.y + 2u;
    T[thread] = T_aux;
    if (threadIdx.x == 0u && xIndex < Lx)
    {
        T[thread - 1u] = T_aux;
    }
    if (threadIdx.x + 1u == blockDim.x)
    {
        T[thread + 1u] = T_aux;
    }
    if (threadIdx.y == 0u)
    {
        T[thread - threadDimX] = T_aux;
    }
    if (threadIdx.y + 1u == blockDim.y)
    {
        T[thread + threadDimX] = T_aux;
    }
    if (threadIdx.z == 0u)
    {
        T[thread - threadDimX * threadDimY] = T_aux;
    }
    if (threadIdx.z + 1u == blockDim.z)
    {
        T[thread + threadDimX * threadDimY] = T_aux;
    }
    __syncthreads();
    diff_aux += DifferentialK(T[thread - 1u], T[thread], T[thread + 1u], dx);
    diff_aux += DifferentialK(T[thread - threadDimX], T[thread], T[thread + threadDimX], dy);
    diff_aux += DifferentialK(T[thread - threadDimX * threadDimY], T[thread], T[thread + threadDimX * threadDimY], dz);

    if (xIndex < Lx && yIndex < Ly && zIndex < Lz)
    {
        diff_out[index] = diff_aux;
    }
}

__global__ void TermalCapacity(double *diff_out, const double *T_in, unsigned Lx, unsigned Ly, unsigned Lz)
{
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned index = GetIndex(xIndex, yIndex, zIndex, Lx, Ly);
    if (xIndex < Lx && yIndex < Ly && zIndex < Lz)
    {
        diff_out[index] /= C(T_in[index]);
    }
}

void HeatConduction::GPU::Differential(double *diff_out, const double *T_in, const double *Q_in, double amp, double dx, double dy, double dz, unsigned Lx, unsigned Ly, unsigned Lz)
{
    // Temperature Diffusion
    dim3 T(8u, 8u, 4u);
    dim3 B((Lx + T.x - 1u) / T.x, (Ly + T.y - 1u) / T.y, (Lz + T.z - 1u) / T.z);
    unsigned size = sizeof(double) * (T.x + 2u) * (T.y + 2u) * (T.z + 2u);
    DifferentialAxis<<<T, B, size, stream>>>(diff_out, T_in, dx, dy, dz, Lx, Ly, Lz);
    // Flux Contribution
    double alpha = amp / dz;
    cublasDaxpy(handle, Lx * Ly, &alpha, Q_in, 1, diff_out + Lx * Ly * (Lz - 1u), 1);
    // Thermal Capacity
    TermalCapacity<<<T, B, size, stream>>>(diff_out, T_in, Lx, Ly, Lz);
}

void HeatConduction::GPU::AllocWorkspaceEuler(double *&workspace_out, unsigned length_in)
{
    cudaMallocAsync(&workspace_out, sizeof(double) * length_in, stream);
}

void HeatConduction::GPU::FreeWorkspaceEuler(double *&workspace_out)
{
    cudaFreeAsync(workspace_out, stream);
}

void HeatConduction::GPU::Euler(double *T_out, double *T_in, double *Q_in, HeatConductionProblem &problem_in, double *workspace)
{
    cublasDcopy(handle, problem_in.Lx * problem_in.Ly * problem_in.Lz, T_in, 1u, T_out, 1u);
    Differential(workspace, T_in, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);
    double alpha = problem_in.dt;
    cublasDaxpy(handle, problem_in.Lx * problem_in.Ly * problem_in.Lz, &alpha, workspace, 1u, T_out, 1u);
}
void HeatConduction::GPU::AllocWorkspaceRK4(double *&workspace_out, unsigned length_in)
{
    cudaMallocAsync(&workspace_out, sizeof(double) * 5u * length_in, stream);
}
void HeatConduction::GPU::FreeWorkspaceRK4(double *&workspace_out)
{
    cudaFreeAsync(workspace_out, stream);
}
void HeatConduction::GPU::RK4(double *T_out, double *T_in, double *Q_in, HeatConductionProblem &problem_in, double *workspace)
{
    double alpha;
    double *k1, *k2, *k3, *k4, *aux;
    unsigned L = problem_in.Lx * problem_in.Ly * problem_in.Lz;

    aux = workspace;
    k1 = aux + L;
    k2 = k1 + L;
    k3 = k2 + L;
    k4 = k3 + L;

    cublasDcopy(handle, L, T_in, 1u, aux, 1u);
    Differential(k1, aux, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);

    cublasDcopy(handle, L, T_in, 1u, aux, 1u);
    alpha = 0.5 * problem_in.dt;
    cublasDaxpy(handle, L, &alpha, k1, 1u, aux, 1u);
    Differential(k2, aux, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);

    cublasDcopy(handle, L, T_in, 1u, aux, 1u);
    alpha = 0.5 * problem_in.dt;
    cublasDaxpy(handle, L, &alpha, k2, 1u, aux, 1u);
    Differential(k3, aux, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);

    cublasDcopy(handle, L, T_in, 1u, aux, 1u);
    alpha = 1.0 * problem_in.dt;
    cublasDaxpy(handle, L, &alpha, k3, 1u, aux, 1u);
    Differential(k4, aux, Q_in, problem_in.amp, problem_in.dx, problem_in.dy, problem_in.dz, problem_in.Lx, problem_in.Ly, problem_in.Lz);

    cublasDcopy(handle, L, T_in, 1u, T_out, 1u);

    cublasDcopy(handle, L, k1, 1u, aux, 1u);
    alpha = 2.0;
    cublasDaxpy(handle, L, &alpha, k2, 1u, aux, 1u);
    cublasDaxpy(handle, L, &alpha, k3, 1u, aux, 1u);
    alpha = 1.0;
    cublasDaxpy(handle, L, &alpha, k4, 1u, aux, 1u);

    alpha = (problem_in.dt / 6.0);
    cublasDaxpy(handle, L, &alpha, aux, 1u, T_out, 1u);
}

__global__ void SetFluxDevice(double *Q_out, double dx, double dy, double Sx, double Sy, unsigned Lx, unsigned Ly)
{
    unsigned xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    bool xCondition, yCondition;
    if (xIndex < Lx && yIndex < Ly)
    {
        xCondition = (xIndex + 0.5) * dx >= 0.4 * Sx && (xIndex + 0.5) * dx <= 0.7 * Sx;
        yCondition = (yIndex + 0.5) * dy >= 0.4 * Sy && (yIndex + 0.5) * dy <= 0.7 * Sy;
        Q_out[yIndex * Lx + xIndex] = xCondition * yCondition * 100.0;
    }
}

void HeatConduction::GPU::SetFlux(double *Q_out, HeatConductionProblem &problem_in, unsigned t_in)
{
    dim3 T(16u, 16u);
    dim3 B((problem_in.Lx + T.x - 1u) / T.x, (problem_in.Ly + T.y - 1u) / T.y);
    SetFluxDevice<<<T, B, 0, stream>>>(Q_out, problem_in.dx, problem_in.dy, problem_in.Sx, problem_in.Sy, problem_in.Lx, problem_in.Ly);
}

void HeatConduction::GPU::AddError(double *T_out, double mean_in, double sigma_in, unsigned length)
{
    Pointer<double> T = Pointer<double>(T_out, PointerType::GPU, PointerContext::GPU_Aware);
    Pointer<double> randomVector = Pointer<double>(PointerType::GPU, PointerContext::GPU_Aware);
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
    curandSetStream(generator, stream);
    curandSetPseudoRandomGeneratorSeed(generator, 1234llu);
    cudaMalloc(&(randomVector.pointer), length);
    curandGenerateNormalDouble(generator, randomVector.pointer, length, mean_in, sigma_in);
    curandDestroyGenerator(generator);
    MathGPU::Add(T, randomVector, length, stream);
}