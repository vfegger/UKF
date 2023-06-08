#include "../include/HeatConductionRadiationCylinder.hpp"

HCRC::HCRCProblem::HCRCProblem() : T0(0.0), Q0(0.0), amp(0.0), r0(0.0), h(0.0), Sr(0.0), Sth(0.0), Sz(0.0), St(0.0), Lr(0u), Lth(0u), Lz(0u), Lt(0u), dr(0.0), dth(0.0), dz(0.0), dt(0.0), iteration(1u)
{
}

HCRC::HCRCProblem::HCRCProblem(double T0_in, double Q0_in, double amp_in, double r0_in, double h_in, double Sr_in, double Sth_in, double Sz_in, double St_in, unsigned Lr_in, unsigned Lth_in, unsigned Lz_in, unsigned Lt_in, unsigned iteration) : T0(T0_in), Q0(Q0_in), amp(amp_in), r0(r0_in), h(h_in), Sr(Sr_in), Sth(Sth_in), Sz(Sz_in), St(St_in), Lr(Lr_in), Lth(Lth_in), Lz(Lz_in), Lt(Lt_in), dr(Sr_in / Lr_in), dth(Sth_in / Lth_in), dz(Sz_in / Lz_in), dt(St_in / (Lt_in * iteration)), iteration(iteration)
{
}

// CPU Section

inline unsigned HCRC::Index3D(unsigned i, unsigned j, unsigned k, unsigned Lr, unsigned Lth, unsigned Lz)
{
    return (k * Lth + j)* Lr + i;
}

inline unsigned HCRC::Index2D(unsigned j, unsigned k, unsigned Lth, unsigned Lz)
{
    return k * Lth + j;
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
    return 1.0;
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
                index = Index3D(i, j, k, Lr, Lth, Lz);
                T0 = T_in[index];
                unsigned iP, jP, kP, iN, jN, kN;
                iP = (i == Lr - 1u) ? Lr - 1u : i + 1u;
                jP = (j == Lth - 1u) ? 0u : j + 1u;
                kP = (k == Lz - 1u) ? Lz - 1u : k + 1u;
                iN = (i == 0u) ? 0u : i - 1u;
                jN = (j == 0u) ? Lth - 1u : j - 1u;
                kN = (k == 0u) ? 0u : k - 1u;
                TiN = T_in[Index3D(iN, j, k, Lr, Lth, Lz)];
                TiP = T_in[Index3D(iP, j, k, Lr, Lth, Lz)];
                TjN = T_in[Index3D(i, jN, k, Lr, Lth, Lz)];
                TjP = T_in[Index3D(i, jP, k, Lr, Lth, Lz)];
                TkN = T_in[Index3D(i, j, kN, Lr, Lth, Lz)];
                TkP = T_in[Index3D(i, j, kP, Lr, Lth, Lz)];
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
            index = Index3D(0u, j, k, Lr, Lth, Lz);
            indexQ = Index2D(j, k, Lth, Lz);
            diff_out[index] += h * (T_amb[0u] - T_in[index]) * r0 * dth * dz;
        }
    }
    // Outside the Cylinder
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            index = Index3D(Lr - 1u, j, k, Lr, Lth, Lz);
            indexQ = Index2D(j, k, Lth, Lz);
            diff_out[index] += h * (T_amb[0u] - T_in[index]) * (r0 + dr * Lr) * dth * dz;
        }
    }
    // Outside the Cylinder
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            index = Index3D(Lr - 1u, j, k, Lr, Lth, Lz);
            indexQ = Index2D(j, k, Lth, Lz);
            diff_out[index] += amp * E(T_in[index]) * Q_in[indexQ];
        }
    }
    // Calculation of the temporal derivative
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            for (unsigned i = 0u; i < Lr; i++)
            {
                index = Index3D(i, j, k, Lr, Lth, Lz);
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
    for (unsigned it = 1u; it < problem_in.iteration; it++)
    {
        Differential(workspace, T_out, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);
        for (unsigned i = 0u; i < L; i++)
        {
            T_out[i] = T_out[i] + problem_in.dt * (workspace[i]);
        }
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

    for (unsigned it = 1u; it < problem_in.iteration; it++)
    {
        Differential(k1, T_out, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

        for (unsigned i = 0u; i < L; i++)
        {
            T_aux[i] = T_out[i] + 0.5 * problem_in.dt * k1[i];
        }
        Differential(k2, T_aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

        for (unsigned i = 0u; i < L; i++)
        {
            T_aux[i] = T_out[i] + 0.5 * problem_in.dt * k2[i];
        }
        Differential(k3, T_aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

        for (unsigned i = 0u; i < L; i++)
        {
            T_aux[i] = T_out[i] + problem_in.dt * k3[i];
        }
        Differential(k4, T_aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

        for (unsigned i = 0u; i < L; i++)
        {
            T_out[i] = T_out[i] + (problem_in.dt / 6.0) * (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]);
        }
    }
}

void HCRC::CPU::SetFlux(double *Q_out, HCRCProblem &problem_in, unsigned t_in)
{
    bool xCondition, yCondition;
    double dth, dz, Sth, Sz;
    unsigned Lth, Lz;
    dth = problem_in.dth;
    dz = problem_in.dz;
    Sth = problem_in.Sth;
    Sz = problem_in.Sz;
    Lth = problem_in.Lth;
    Lz = problem_in.Lz;
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            xCondition = (j + 0.5) * dth >= 0.4 * Sth && (j + 0.5) * dth <= 0.7 * Sth;
            yCondition = (k + 0.5) * dz >= 0.4 * Sz && (k + 0.5) * dz <= 0.7 * Sz;
            Q_out[k * Lth + j] = (xCondition && yCondition) ? 1000.0 : 0.0;
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
    return 1.0;
}

__device__ double DifferentialK(const double T0_in, const double TP_in, const double delta_in, const double coef_in)
{
    double K_mean = 2.0 * (K(TP_in) * K(T0_in)) / (K(TP_in) + K(T0_in));
    double diff = (TP_in - T0_in) / delta_in;
    return coef_in * K_mean * diff;
}

__device__ inline int clamp(int i, int min, int max)
{
    const int t = (i < min) ? min : i;
    return (t > max) ? max : t;
}

__device__ inline int Index3D_dev(int i, int j, int k, int Li, int Lj, int Lk)
{
    return (k * Lj + j) * Li + i;
}

__device__ inline int Index2D_dev(int j, int k, int Lj, int Lk)
{
    return k * Lj + j;
}

__global__ void DifferentialAxis(double *diff_out, const double *T_in, double r0, double dr, double dth, double dz, int Lr, int Lth, int Lz)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    int index = Index3D_dev(xIndex, yIndex, zIndex, Lr, Lth, Lz);
    extern __shared__ double T[];
    int thread;
    int threadDimX, threadDimY, threadDimZ;
    int threadStrideX, threadStrideY, threadStrideZ;
    double T_aux = 0.0, diff_aux = 0.0;
    threadDimX = blockDim.x + 2;
    threadDimY = blockDim.y + 2;
    threadDimZ = blockDim.z + 2;
    threadStrideX = 1;
    threadStrideY = threadDimX;
    threadStrideZ = threadDimX * threadDimY;
    thread = Index3D_dev(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1, threadDimX, threadDimY, threadDimZ);
    
    bool inside = xIndex < Lr && yIndex < Lth && zIndex < Lz;
    if (inside)
    {
        T_aux = T_in[index];
    }
    T[thread] = T_aux;
    __syncthreads();

    if (threadIdx.x == 0 && inside)
    {
        T[thread - threadStrideX] = T_in[Index3D_dev(max(xIndex - 1, 0), yIndex, zIndex, Lr, Lth, Lz)];
    }
    if (threadIdx.x + 1 == blockDim.x || (xIndex + 1u == Lr && inside))
    {
        T[thread + threadStrideX] = T_in[Index3D_dev(min(xIndex + 1, 0), yIndex, zIndex, Lr, Lth, Lz)];
    }

    if (threadIdx.y == 0 && inside)
    {
        T[thread - threadStrideY] = T_in[Index3D_dev(xIndex, (yIndex == 0) * (Lth - 1) + (yIndex != 0) * (yIndex - 1), zIndex, Lr, Lth, Lz)];
    }
    if (threadIdx.y + 1 == blockDim.y || (yIndex + 1u == Lth && inside))
    {
        T[thread + threadStrideY] = T_in[Index3D_dev(xIndex, (yIndex + 1 != Lth) * (yIndex + 1), zIndex, Lr, Lth, Lz)];
    }

    if (threadIdx.z == 0 && inside)
    {
        T[thread - threadStrideZ] = T_in[Index3D_dev(xIndex, yIndex, max(zIndex - 1, 0), Lr, Lth, Lz)];
    }
    if (threadIdx.z + 1 == blockDim.z || (zIndex + 1 == Lz && inside))
    {
        T[thread + threadStrideZ] = T_in[Index3D_dev(xIndex, yIndex, min(zIndex + 1, 0), Lr, Lth, Lz)];
    }
    __syncthreads();

    diff_aux += DifferentialK(T[thread], T[thread - threadStrideX], dr, (r0 + dr * xIndex) * dth * dz);
    diff_aux += DifferentialK(T[thread], T[thread + threadStrideX], dr, (r0 + dr * (xIndex + 1)) * dth * dz);
    diff_aux += DifferentialK(T[thread], T[thread - threadStrideY], dth * (r0 + dr * xIndex), dr * dz);
    diff_aux += DifferentialK(T[thread], T[thread + threadStrideY], dth * (r0 + dr * (xIndex + 1)), dr * dz);
    diff_aux += DifferentialK(T[thread], T[thread - threadStrideZ], dz, dr * (r0 + dr * xIndex) * dth);
    diff_aux += DifferentialK(T[thread], T[thread + threadStrideZ], dz, dr * (r0 + dr * (xIndex + 1)) * dth);

    if (inside)
    {
        diff_out[index] = diff_aux;
    }
}

__global__ void FluxContribution(double *diff_out, const double *T_in, const double *Q_in, const double *T_amb, double amp, double r0, double h, double dr, double dth, double dz, int Lr, int Lth, int Lz)
{
    int yIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int zIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int indexRinf = Index3D_dev(0u, yIndex, zIndex, Lr, Lth, Lz);
    int indexRsup = Index3D_dev(Lr - 1u, yIndex, zIndex, Lr, Lth, Lz);
    int indexQ = Index2D_dev(yIndex, zIndex, Lth, Lz);
    if (yIndex < Lth && zIndex < Lz)
    {
        diff_out[indexRinf] += h * (T_amb[0u] - T_in[indexRinf]) * (r0 + dr * Lr) * dth * dz;
        diff_out[indexRsup] += h * (T_amb[0u] - T_in[indexRsup]) * (r0 + dr * Lr) * dth * dz;
        diff_out[indexRsup] += amp * E(T_in[indexRsup]) * Q_in[indexQ];
    }
}

__global__ void TermalCapacity(double *diff_out, const double *T_in, double r0, double dr, double dth, double dz, int Lr, int Lth, int Lz)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    int index = Index3D_dev(xIndex, yIndex, zIndex, Lr, Lth, Lz);
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
    DifferentialAxis<<<B, T, size, stream_in>>>(diff_out, T_in, r0, dr, dth, dz, (int)Lr, (int)Lth, (int)Lz);
    // Flux Contribution
    dim3 Tq(16u, 16u);
    dim3 Bq((Lth + T.y - 1u) / T.y, (Lz + T.z - 1u) / T.z);
    size = 0u;
    FluxContribution<<<Bq, Tq, size, stream_in>>>(diff_out, T_in, Q_in, T_amb_in, amp, r0, h, dr, dth, dz, (int)Lr, (int)Lth, (int)Lz);
    // Thermal Capacity
    size = 0u;
    TermalCapacity<<<B, T, size, stream_in>>>(diff_out, T_in, r0, dr, dth, dz, (int)Lr, (int)Lth, (int)Lz);
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
    for (unsigned it = 1u; it < problem_in.iteration; it++)
    {
        Differential(workspace, T_out, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);
        cublasDaxpy(handle_in, problem_in.Lr * problem_in.Lth * problem_in.Lz, &alpha, workspace, 1u, T_out, 1u);
    }
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

    for (unsigned it = 1u; it < problem_in.iteration; it++)
    {
        cublasDcopy(handle_in, L, T_out, 1u, aux, 1u);
        Differential(k1, aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

        cublasDcopy(handle_in, L, T_out, 1u, aux, 1u);
        alpha = 0.5 * problem_in.dt;
        cublasDaxpy(handle_in, L, &alpha, k1, 1u, aux, 1u);
        Differential(k2, aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

        cublasDcopy(handle_in, L, T_out, 1u, aux, 1u);
        alpha = 0.5 * problem_in.dt;
        cublasDaxpy(handle_in, L, &alpha, k2, 1u, aux, 1u);
        Differential(k3, aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

        cublasDcopy(handle_in, L, T_out, 1u, aux, 1u);
        alpha = 1.0 * problem_in.dt;
        cublasDaxpy(handle_in, L, &alpha, k3, 1u, aux, 1u);
        Differential(k4, aux, Q_in, T_amb_in, problem_in.amp, problem_in.r0, problem_in.h, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

        cublasDcopy(handle_in, L, k1, 1u, aux, 1u);
        alpha = 2.0;
        cublasDaxpy(handle_in, L, &alpha, k2, 1u, aux, 1u);
        cublasDaxpy(handle_in, L, &alpha, k3, 1u, aux, 1u);
        alpha = 1.0;
        cublasDaxpy(handle_in, L, &alpha, k4, 1u, aux, 1u);

        alpha = (problem_in.dt / 6.0);
        cublasDaxpy(handle_in, L, &alpha, aux, 1u, T_out, 1u);
    }
}

__global__ void SetFluxDevice(double *Q_out, double dth, double dz, double Sth, double Sz, unsigned Lth, unsigned Lz)
{
    unsigned xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    bool xCondition, yCondition;
    if (xIndex < Lth && yIndex < Lz)
    {
        xCondition = (xIndex + 0.5) * dth >= 0.4 * Sth && (xIndex + 0.5) * dth <= 0.7 * Sz;
        yCondition = (yIndex + 0.5) * dz >= 0.4 * Sz && (yIndex + 0.5) * dz <= 0.7 * Sz;
        Q_out[yIndex * Lth + xIndex] = xCondition * yCondition * 1000.0;
    }
}

void HCRC::GPU::SetFlux(double *Q_out, HCRCProblem &problem_in, unsigned t_in, cudaStream_t stream_in)
{
    dim3 T(16u, 16u);
    dim3 B((problem_in.Lth + T.x - 1u) / T.x, (problem_in.Lz + T.y - 1u) / T.y);
    SetFluxDevice<<<B, T, 0, stream_in>>>(Q_out, problem_in.dth, problem_in.dz, problem_in.Sth, problem_in.Sz, problem_in.Lth, problem_in.Lz);
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