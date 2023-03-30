#include "../include/HeatConductionRadiationCylinder.hpp"

HCRC::HCRCProblem::HCRCProblem() : T0(0.0), Q0(0.0), amp(0.0), r0(0.0), h(0.0), Sr(0.0), Sth(0.0), Sz(0.0), St(0.0), Lr(0u), Lth(0u), Lz(0u), Lt(0u), dr(0.0), dth(0.0), dz(0.0), dt(0.0)
{
}

HCRC::HCRCProblem::HCRCProblem(double T0_in, double Q0_in, double amp_in, double r0_in, double h_in, double Sr_in, double Sth_in, double Sz_in, double St_in, unsigned Lr_in, unsigned Lth_in, unsigned Lz_in, unsigned Lt_in) : T0(T0_in), Q0(Q0_in), amp(amp_in), r0(r0_in), h(h_in), Sr(Sr_in - r0_in), Sth(Sth_in), Sz(Sz_in), St(St_in), Lr(Lr_in), Lth(Lth_in), Lz(Lz_in), Lt(Lt_in), dr((Sr_in - r0_in) / Lr_in), dth(Sth_in / Lth_in), dz(Sz_in / Lz_in), dt(St_in / Lt_in)
{
}

// CPU Section

inline unsigned Index(unsigned i, unsigned j, unsigned k, unsigned Lr, unsigned Lth, unsigned Lz)
{
    return (std::clamp(k, 0u, Lz - 1) * Lth + (j % Lth)) * Lr + std::clamp(i, 0u, Lr - 1u);
}

inline unsigned Index(unsigned j, unsigned k, unsigned Lth, unsigned Lz)
{
    return std::clamp(k, 0u, Lz - 1) * Lth + (j % Lth);
}

inline double HCRC::CPU::C(double T_in)
{
    return 1324.75 * T_in + 3557900.0;
}

inline double HCRC::CPU::K(double T_in)
{
    return (14e-3 + 2.517e-6 * T_in) * T_in + 12.45;
}

inline double HCRC::CPU::E(double T_in)
{
    return 0.5;
}

double HCRC::CPU::DifferentialK(const double T0_in, const double TP_in, const double delta_in, const double coef_in)
{
    double K_mean = 2.0 * (K(TP_in) * K(T0_in)) / (K(TP_in) + K(T0_in));
    double diff = (TP_in - T0_in) / delta_in;
    return coef_in * K_mean * diff;
}

void HCRC::CPU::Differential(double *diff_out, const double *T_in, const double *Q_in, const double *T_amb, double amp, double r0, double h, double dr, double dth, double dz, unsigned Lr, unsigned Lth, unsigned Lz)
{
    unsigned index, indexQ;
    double acc;
    double T0, TiN, TiP, TjN, TjP, TkN, TkP;
    // Diffusion Surfaces
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            for (unsigned i = 0u; i < Lr; i++)
            {
                index = Index(i, j, k, Lr, Lth, Lz);
                T0 = T_in[index];
                TiN = T_in[Index(i - 1, j, k, Lr, Lth, Lz)];
                TiP = T_in[Index(i + 1, j, k, Lr, Lth, Lz)];
                TjN = T_in[Index(i, j - 1, k, Lr, Lth, Lz)];
                TjP = T_in[Index(i, j + 1, k, Lr, Lth, Lz)];
                TkN = T_in[Index(i, j, k - 1, Lr, Lth, Lz)];
                TkP = T_in[Index(i, j, k + 1, Lr, Lth, Lz)];
                acc = 0.0;
                // R dependency
                double R_N = DifferentialK(T0, TiN, dr, (r0 + dr * i) * dth * dz);
                double R_P = DifferentialK(T0, TiP, dr, (r0 + dr * (i + 1u)) * dth * dz);
                acc += R_N + R_P;
                // Th dependency
                double Th_N = DifferentialK(T0, TjN, dth * (r0 + dr * i), dr * dz);
                double Th_P = DifferentialK(T0, TjP, dth * (r0 + dr * (i + 1u)), dr * dz);
                acc += Th_N + Th_P;
                // Z dependency
                double Z_N = DifferentialK(T0, TkN, dz, dr * (r0 + dr * i) * dth);
                double Z_P = DifferentialK(T0, TkP, dz, dr * (r0 + dr * (i + 1u)) * dth);
                acc += Z_N + Z_P;

                diff_out[index] = acc;
            }
        }
    }
    // Influx Surfaces (Countour Region)
    // Inside the Cylinder
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            index = Index(0u, j, k, Lr, Lth, Lz);
            indexQ = Index(j, k, Lth, Lz);
            diff_out[index] += h * (T_amb[0u] - T_in[index]) * r0 * dth * dz;
        }
    }
    // Outside the Cylinder
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            index = Index(Lr - 1u, j, k, Lr, Lth, Lz);
            indexQ = Index(j, k, Lth, Lz);
            diff_out[index] += h * (T_amb[0u] - T_in[index]) * (r0 + dr * Lr) * dth * dz;
        }
    }
    // Outside the Cylinder
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            index = Index(Lr - 1u, j, k, Lr, Lth, Lz);
            indexQ = Index(j, k, Lth, Lz);
            diff_out[index] += amp * E(T_in[index]) * Q_in[indexQ] * (r0 + dr * Lr) * dth * dz;
        }
    }
    // Calculation of the temporal derivative
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            for (unsigned i = 0u; i < Lr; i++)
            {
                index = Index(i, j, Lz - 1u, Lr, Lth, Lz);
                double mR = r0 + dr * (i + 0.5);
                double vol = dr * mR * dth * dz;
                diff_out[index] /= vol * C(T_in[index]);
            }
        }
    }
}

void HCRC::CPU::AllocWorkspaceEuler(double *&workspace_out, unsigned length_in)
{
    workspace_out = new double[length_in];
}

void HCRC::CPU::FreeWorkspaceEuler(double *&workspace_in)
{
    delete[] workspace_in;
    workspace_in = NULL;
}

void HCRC::CPU::Euler(double *T_out, const double *T_in, const double *Q_in, const double *T_amb_in, HCRCProblem &problem_in, double *workspace)
{
    unsigned L = problem_in.Lr * problem_in.Lth * problem_in.Lz;
    Differential(workspace, T_in, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);
    for (unsigned i = 0u; i < L; i++)
    {
        T_out[i] = T_in[i] + problem_in.dt * (workspace[i]);
    }
}

void HCRC::CPU::AllocWorkspaceRK4(double *&workspace_out, unsigned length_in)
{
    workspace_out = new double[5u * length_in];
}

void HCRC::CPU::FreeWorkspaceRK4(double *&workspace_in)
{
    delete[] workspace_in;
    workspace_in = NULL;
}

void HCRC::CPU::RK4(double *T_out, const double *T_in, const double *Q_in, const double *T_amb_in, HCRCProblem &problem_in, double *workspace)
{
    unsigned L = problem_in.Lr * problem_in.Lth * problem_in.Lz;
    double *k1, *k2, *k3, *k4;
    double *T_aux;
    T_aux = workspace;
    k1 = T_aux + L;
    k2 = k1 + L;
    k3 = k2 + L;
    k4 = k3 + L;
    Differential(k1, T_in, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i] + 0.5 * problem_in.dt * k1[i];
    }
    Differential(k2, T_aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i] + 0.5 * problem_in.dt * k2[i];
    }
    Differential(k3, T_aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i] + problem_in.dt * k3[i];
    }
    Differential(k4, T_aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_out[i] = T_in[i] + (problem_in.dt / 6.0) * (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]);
    }
}

void HCRC::CPU::SetFlux(double *Q_out, HCRCProblem &problem_in, unsigned t_in)
{
    bool xCondition, yCondition;
    double dr, dth, Sr, Sth;
    unsigned Lr, Lth;
    dr = problem_in.dr;
    dth = problem_in.dth;
    Sr = problem_in.Sr;
    Sth = problem_in.Sth;
    Lr = problem_in.Lr;
    Lth = problem_in.Lth;
    for (unsigned j = 0u; j < Lth; j++)
    {
        for (unsigned i = 0u; i < Lr; i++)
        {
            xCondition = (i + 0.5) * dr >= 0.4 * Sr && (i + 0.5) * dr <= 0.7 * Sr;
            yCondition = (j + 0.5) * dth >= 0.4 * Sth && (j + 0.5) * dth <= 0.7 * Sth;
            Q_out[j * Lr + i] = (xCondition && yCondition) ? 100.0 : 0.0;
        }
    }
}

void HCRC::CPU::AddError(double *T_out, double mean_in, double sigma_in, unsigned length)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean_in, sigma_in);
    for (unsigned i = 0u; i < length; i++)
    {
        T_out[i] += distribution(generator);
    }
}

void HCRC::CPU::SelectTemperatures(double *T_out, double *T_in, unsigned *indexR_in, unsigned *indexTh_in, unsigned *indexZ_in, unsigned length_in, unsigned Lr, unsigned Lth, unsigned Lz)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        T_out[i] = T_in[Index(indexR_in[i], indexTh_in[i], indexZ_in[i], Lr, Lth, Lz)];
    }
}

// GPU Section

__device__ inline double C(double T_in)
{
    return 1324.75 * T_in + 3557900.0;
}

__device__ inline double K(double T_in)
{
    return (14e-3 + 2.517e-6 * T_in) * T_in + 12.45;
}

__device__ inline double E(double T_in)
{
    return 0.5;
}

__device__ double DifferentialK(const double T0_in, const double TP_in, const double delta_in, const double coef_in)
{
    double K_mean = 2.0 * (K(TP_in) * K(T0_in)) / (K(TP_in) + K(T0_in));
    double diff = (TP_in - T0_in) / delta_in;
    return coef_in * K_mean * diff;
}

__device__ inline unsigned clamp(unsigned i, unsigned min, unsigned max)
{
    const unsigned t = (i < min) ? min : i;
    return (t > max) ? max : t;
}

__device__ inline unsigned Index3D(unsigned i, unsigned j, unsigned k, unsigned Li, unsigned Lj, unsigned Lk)
{
    return (clamp(k, 0u, Lk - 1u) * Lj + (j % Lj)) * Li + clamp(i, 0u, Li - 1u);
}

__device__ inline unsigned IndexThread(unsigned i, unsigned j, unsigned k, unsigned Li, unsigned Lj)
{
    return (k * Lj + j) * Li + i;
}

__device__ inline unsigned Index2D(unsigned j, unsigned k, unsigned Lj, unsigned Lk)
{
    return clamp(k, 0u, Lk - 1u) * Lj + (j % Lj);
}

__global__ void DifferentialAxis(double *diff_out, const double *T_in, double r0, double dr, double dth, double dz, unsigned Lr, unsigned Lth, unsigned Lz)
{
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned index = Index3D(xIndex, yIndex, zIndex, Lr, Lth, Lz);
    extern __shared__ double T[];
    unsigned thread;
    unsigned threadDimX;
    unsigned threadDimY;
    double T_aux = 0.0;
    double diff_aux = 0.0;
    bool inside = xIndex < Lr && yIndex < Lth && zIndex < Lz;
    if (inside)
    {
        T_aux = T_in[index];
    }
    threadDimX = blockDim.x + 2u;
    threadDimY = blockDim.y + 2u;
    thread = IndexThread(threadIdx.x + 1u, threadIdx.y + 1u, threadIdx.z + 1u, threadDimX, threadDimY);
    T[thread] = T_aux;
    __syncthreads();
    if (threadIdx.x == 0u && inside)
    {
        T[thread - 1u] = T_in[Index3D(xIndex - 1u, yIndex, zIndex, Lr, Lth, Lz)];
    }
    if (threadIdx.x + 1u == blockDim.x || xIndex + 1u == Lr && inside)
    {
        T[thread + 1u] = T_in[Index3D(xIndex + 1u, yIndex, zIndex, Lr, Lth, Lz)];
    }

    if (threadIdx.y == 0u && inside)
    {
        T[thread - threadDimX] = T_in[Index3D(xIndex, yIndex - 1u, zIndex, Lr, Lth, Lz)];
    }
    if (threadIdx.y + 1u == blockDim.y || yIndex + 1u == Lth && inside)
    {
        T[thread + threadDimX] = T_in[Index3D(xIndex, yIndex + 1u, zIndex, Lr, Lth, Lz)];
    }
    if (threadIdx.z == 0u && inside)
    {
        T[thread - threadDimX * threadDimY] = T_in[Index3D(xIndex, yIndex, zIndex - 1u, Lr, Lth, Lz)];
    }
    if (threadIdx.z + 1u == blockDim.z || zIndex + 1u == Lz && inside)
    {
        T[thread + threadDimX * threadDimY] = T_in[Index3D(xIndex, yIndex, zIndex + 1u, Lr, Lth, Lz)];
    }
    __syncthreads();
    diff_aux += DifferentialK(T[thread], T[thread - 1u], dr, (r0 + dr * xIndex) * dth * dz);
    diff_aux += DifferentialK(T[thread], T[thread + 1u], dr, (r0 + dr * (xIndex + 1u)) * dth * dz);
    diff_aux += DifferentialK(T[thread], T[thread - threadDimX], dth * (r0 + dr * xIndex), dr * dz);
    diff_aux += DifferentialK(T[thread], T[thread + threadDimX], dth * (r0 + dr * (xIndex + 1u)), dr * dz);
    diff_aux += DifferentialK(T[thread], T[thread - threadDimX * threadDimY], dz, dr * (r0 + dr * xIndex) * dth);
    diff_aux += DifferentialK(T[thread], T[thread + threadDimX * threadDimY], dz, dr * (r0 + dr * (xIndex + 1u)) * dth);

    if (inside)
    {
        diff_out[index] = diff_aux;
    }
}

__global__ void FluxContribution(double *diff_out, const double *T_in, const double *Q_in, const double *T_amb, double amp, double r0, double h, double dr, double dth, double dz, unsigned Lr, unsigned Lth, unsigned Lz)
{
    unsigned yIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned zIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned indexRsup = Index3D(Lr - 1u, yIndex, zIndex, Lr, Lth, Lz);
    unsigned indexRinf = Index3D(0u, yIndex, zIndex, Lr, Lth, Lz);
    unsigned indexQ = Index2D(yIndex, zIndex, Lth, Lz);
    if (yIndex < Lth && zIndex < Lz)
    {
        diff_out[indexRsup] += h * (T_amb[0u] - T_in[indexRinf]) * (r0 + dr * Lr) * dth * dz;
        diff_out[indexRsup] += h * (T_amb[0u] - T_in[indexRsup]) * (r0 + dr * Lr) * dth * dz;
        diff_out[indexRsup] += amp * E(T_in[indexRsup]) * Q_in[indexQ] * (r0 + dr * Lr) * dth * dz;
    }
}

__global__ void TermalCapacity(double *diff_out, const double *T_in, double r0, double dr, double dth, double dz, unsigned Lr, unsigned Lth, unsigned Lz)
{
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned index = Index3D(xIndex, yIndex, zIndex, Lr, Lth, Lz);
    if (xIndex < Lr && yIndex < Lth && zIndex < Lz)
    {
        double mR = r0 + dr * (xIndex + 0.5);
        double vol = dr * mR * dth * dz;
        diff_out[index] /= vol * C(T_in[index]);
    }
}

void HCRC::GPU::Differential(double *diff_out, const double *T_in, const double *Q_in, const double *T_amb_in, double amp, double r0, double h, double dr, double dth, double dz, unsigned Lr, unsigned Lth, unsigned Lz, cublasHandle_t handle_in, cudaStream_t stream_in)
{
    cublasSetStream(handle_in, stream_in);
    // Temperature Diffusion
    dim3 T(8u, 8u, 4u);
    dim3 B((Lr + T.x - 1u) / T.x, (Lth + T.y - 1u) / T.y, (Lz + T.z - 1u) / T.z);
    unsigned size = sizeof(double) * (T.x + 2u) * (T.y + 2u) * (T.z + 2u);
    DifferentialAxis<<<B, T, size, stream_in>>>(diff_out, T_in, r0, dr, dth, dz, Lr, Lth, Lz);
    // Flux Contribution
    dim3 Tq(16u, 16u);
    dim3 Bq((Lth + T.y - 1u) / T.y, (Lz + T.z - 1u) / T.z);
    FluxContribution<<<Bq, Tq, size, stream_in>>>(diff_out, T_in, Q_in, T_amb_in, amp, r0, h, dr, dth, dz, Lr, Lth, Lz);
    // Thermal Capacity
    TermalCapacity<<<B, T, size, stream_in>>>(diff_out, T_in, r0, dr, dth, dz, Lr, Lth, Lz);
}

void HCRC::GPU::AllocWorkspaceEuler(double *&workspace_out, unsigned length_in, cudaStream_t stream_in)
{
    cudaMallocAsync(&workspace_out, sizeof(double) * length_in, stream_in);
}

void HCRC::GPU::FreeWorkspaceEuler(double *&workspace_out, cudaStream_t stream_in)
{
    cudaFreeAsync(workspace_out, stream_in);
}

void HCRC::GPU::Euler(double *T_out, double *T_in, double *Q_in, double *T_amb_in, HCRCProblem &problem_in, double *workspace, cublasHandle_t handle_in, cudaStream_t stream_in)
{
    cublasDcopy(handle_in, problem_in.Lr * problem_in.Lth * problem_in.Lz, T_in, 1u, T_out, 1u);
    Differential(workspace, T_in, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);
    double alpha = problem_in.dt;
    cublasDaxpy(handle_in, problem_in.Lr * problem_in.Lth * problem_in.Lz, &alpha, workspace, 1u, T_out, 1u);
}
void HCRC::GPU::AllocWorkspaceRK4(double *&workspace_out, unsigned length_in, cudaStream_t stream_in)
{
    cudaMallocAsync(&workspace_out, sizeof(double) * 5u * length_in, stream_in);
}
void HCRC::GPU::FreeWorkspaceRK4(double *&workspace_out, cudaStream_t stream_in)
{
    cudaFreeAsync(workspace_out, stream_in);
}
void HCRC::GPU::RK4(double *T_out, double *T_in, double *Q_in, double *T_amb_in, HCRCProblem &problem_in, double *workspace, cublasHandle_t handle_in, cudaStream_t stream_in)
{
    cublasSetStream(handle_in, stream_in);
    double alpha;
    double *k1, *k2, *k3, *k4, *aux;
    unsigned L = problem_in.Lr * problem_in.Lth * problem_in.Lz;

    aux = workspace;
    k1 = aux + L;
    k2 = k1 + L;
    k3 = k2 + L;
    k4 = k3 + L;

    cublasDcopy(handle_in, L, T_in, 1u, aux, 1u);
    Differential(k1, aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

    cublasDcopy(handle_in, L, T_in, 1u, aux, 1u);
    alpha = 0.5 * problem_in.dt;
    cublasDaxpy(handle_in, L, &alpha, k1, 1u, aux, 1u);
    Differential(k2, aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

    cublasDcopy(handle_in, L, T_in, 1u, aux, 1u);
    alpha = 0.5 * problem_in.dt;
    cublasDaxpy(handle_in, L, &alpha, k2, 1u, aux, 1u);
    Differential(k3, aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

    cublasDcopy(handle_in, L, T_in, 1u, aux, 1u);
    alpha = 1.0 * problem_in.dt;
    cublasDaxpy(handle_in, L, &alpha, k3, 1u, aux, 1u);
    Differential(k4, aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

    cublasDcopy(handle_in, L, T_in, 1u, T_out, 1u);

    cublasDcopy(handle_in, L, k1, 1u, aux, 1u);
    alpha = 2.0;
    cublasDaxpy(handle_in, L, &alpha, k2, 1u, aux, 1u);
    cublasDaxpy(handle_in, L, &alpha, k3, 1u, aux, 1u);
    alpha = 1.0;
    cublasDaxpy(handle_in, L, &alpha, k4, 1u, aux, 1u);

    alpha = (problem_in.dt / 6.0);
    cublasDaxpy(handle_in, L, &alpha, aux, 1u, T_out, 1u);
}

__global__ void SetFluxDevice(double *Q_out, double dr, double dth, double Sr, double Sth, unsigned Lr, unsigned Lth)
{
    unsigned xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    bool xCondition, yCondition;
    if (xIndex < Lr && yIndex < Lth)
    {
        xCondition = (xIndex + 0.5) * dr >= 0.4 * Sr && (xIndex + 0.5) * dr <= 0.7 * Sr;
        yCondition = (yIndex + 0.5) * dth >= 0.4 * Sth && (yIndex + 0.5) * dth <= 0.7 * Sth;
        Q_out[yIndex * Lr + xIndex] = xCondition * yCondition * 100.0;
    }
}

void HCRC::GPU::SetFlux(double *Q_out, HCRCProblem &problem_in, unsigned t_in, cudaStream_t stream_in)
{
    dim3 T(16u, 16u);
    dim3 B((problem_in.Lr + T.x - 1u) / T.x, (problem_in.Lth + T.y - 1u) / T.y);
    SetFluxDevice<<<B, T, 0, stream_in>>>(Q_out, problem_in.dr, problem_in.dth, problem_in.Sr, problem_in.Sth, problem_in.Lr, problem_in.Lth);
}

void HCRC::GPU::AddError(double *T_out, double mean_in, double sigma_in, unsigned length, cudaStream_t stream_in)
{
    Pointer<double> T = Pointer<double>(T_out, PointerType::GPU, PointerContext::GPU_Aware);
    Pointer<double> randomVector = Pointer<double>(PointerType::GPU, PointerContext::GPU_Aware);
    cudaMalloc(&(randomVector.pointer), sizeof(double) * length);
    curandStatus_t status;
    curandGenerator_t generator;
    status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
    if (status != curandStatus_t::CURAND_STATUS_SUCCESS)
    {
        std::cout << "GPU curand error: " << status << "\n";
    }
    status = curandSetStream(generator, stream_in);
    if (status != curandStatus_t::CURAND_STATUS_SUCCESS)
    {
        std::cout << "GPU curand error: " << status << "\n";
    }
    status = curandSetPseudoRandomGeneratorSeed(generator, 1234llu);
    if (status != curandStatus_t::CURAND_STATUS_SUCCESS)
    {
        std::cout << "GPU curand error: " << status << "\n";
    }
    status = curandGenerateNormalDouble(generator, randomVector.pointer, length, mean_in, sigma_in);
    if (status != curandStatus_t::CURAND_STATUS_SUCCESS)
    {
        std::cout << "GPU curand error: " << status << "\n";
    }
    status = curandDestroyGenerator(generator);
    if (status != curandStatus_t::CURAND_STATUS_SUCCESS)
    {
        std::cout << "GPU curand error: " << status << "\n";
    }
    MathGPU::Add(T, randomVector, length, stream_in);
    cudaStreamSynchronize(stream_in);
    cudaFree(randomVector.pointer);
}

void HCRC::GPU::SelectTemperatures(double *T_out, double *T_in, unsigned *indexR_in, unsigned *indexTh_in, unsigned *indexZ_in, unsigned length_in, unsigned Lr, unsigned Lth, unsigned Lz)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        cudaMemcpy(T_out + i, T_in + Index(indexR_in[i], indexTh_in[i], indexZ_in[i], Lr, Lth, Lz), sizeof(double), cudaMemcpyDeviceToDevice);
    }
}