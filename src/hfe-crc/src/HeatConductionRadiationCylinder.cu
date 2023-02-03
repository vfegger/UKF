#include "../include/HeatConductionRadiationCylinder.hpp"

HeatConductionRadiationCylinder::HeatConductionRadiationCylinderProblem::HeatConductionRadiationCylinderProblem() : T0(0.0), Q0(0.0), amp(0.0), r0(0.0), Sr(0.0), Sth(0.0), Sz(0.0), St(0.0), Lr(0u), Lth(0u), Lz(0u), Lt(0u), dr(0.0), dth(0.0), dz(0.0), dt(0.0)
{
}

HeatConductionRadiationCylinder::HeatConductionRadiationCylinderProblem::HeatConductionRadiationCylinderProblem(double T0_in, double Q0_in, double amp_in, double r0_in, double Sr_in, double Sth_in, double Sz_in, double St_in, unsigned Lr_in, unsigned Lth_in, unsigned Lz_in, unsigned Lt_in) : T0(T0_in), Q0(Q0_in), amp(amp_in), r0(r0_in), Sr(Sr_in), Sth(Sth_in), Sz(Sz_in), St(St_in), Lr(Lr_in), Lth(Lth_in), Lz(Lz_in), Lt(Lt_in), dr(Sr_in / Lr_in), dth(Sth_in / Lth_in), dz(Sz_in / Lz_in), dt(St_in / Lt_in)
{
}

// CPU Section

inline unsigned Index(unsigned i, unsigned j, unsigned k, unsigned Lr, unsigned Lth, unsigned Lz)
{
    return (std::clamp(k, 0u, Lz - 1) * Lth + (j % Lth)) * Lr + std::clamp(i, 0u, Lr - 1u);
}

inline unsigned Index(unsigned i, unsigned j, unsigned Lr, unsigned Lth)
{
    return (j % Lth) * Lr + std::clamp(i, 0u, Lr);
}

inline double HeatConductionRadiationCylinder::CPU::C(double T_in)
{
    return 1324.75 * T_in + 3557900.0;
}

inline double HeatConductionRadiationCylinder::CPU::K(double T_in)
{
    return (14e-3 + 2.517e-6 * T_in) * T_in + 12.45;
}

inline double HeatConductionRadiationCylinder::CPU::E(double T_in)
{
    return 0.5;
}

double HeatConductionRadiationCylinder::CPU::DifferentialK(double TN_in, double T_in, double TP_in, double delta_in, double coefN, double coefP)
{
    double auxN = 2.0 * (K(TN_in) * K(T_in)) / (K(TN_in) + K(T_in)) * (TN_in - T_in) / delta_in;
    double auxP = 2.0 * (K(TP_in) * K(T_in)) / (K(TP_in) + K(T_in)) * (TP_in - T_in) / delta_in;
    return (coefN * auxN + coefP * auxP) / delta_in;
}

void HeatConductionRadiationCylinder::CPU::Differential(double *diff_out, const double *T_in, const double *Q_in, double amp, double r0, double dr, double dth, double dz, unsigned Lr, unsigned Lth, unsigned Lz)
{
    unsigned index, indexQ;
    double acc;
    double T0, TiN, TiP, TjN, TjP, TkN, TkP;
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
                // X dependency
                acc += DifferentialK(TiN, T0, TiP, dr, 1.0, 1.0);
                // Y dependency
                acc += DifferentialK(TjN, T0, TjP, dth, 1.0 / (r0 - (dr * Lr) / 2.0), 1.0 / (r0 + (dr * Lr) / 2.0));
                // Z dependency
                acc += DifferentialK(TkN, T0, TkP, dz, 1.0, 1.0);

                diff_out[index] = acc;
            }
        }
    }
    for (unsigned j = 0u; j < Lth; j++)
    {
        for (unsigned i = 0u; i < Lr; i++)
        {
            index = Index(i, j, Lz - 1u, Lr, Lth, Lz);
            indexQ = Index(i, j, Lr, Lth);
            diff_out[index] += (amp / dz) * E(T_in[index]) * Q_in[indexQ];
        }
    }
    for (unsigned k = 0u; k < Lz; k++)
    {
        for (unsigned j = 0u; j < Lth; j++)
        {
            for (unsigned i = 0u; i < Lr; i++)
            {
                index = Index(i, j, Lz - 1u, Lr, Lth, Lz);
                diff_out[index] /= C(T_in[index]);
            }
        }
    }
}

void HeatConductionRadiationCylinder::CPU::AllocWorkspaceEuler(double *&workspace_out, unsigned length_in)
{
    workspace_out = new double[length_in];
}

void HeatConductionRadiationCylinder::CPU::FreeWorkspaceEuler(double *&workspace_in)
{
    delete[] workspace_in;
    workspace_in = NULL;
}

void HeatConductionRadiationCylinder::CPU::Euler(double *T_out, const double *T_in, const double *Q_in, HeatConductionRadiationCylinderProblem &problem_in, double *workspace)
{
    unsigned L = problem_in.Lr * problem_in.Lth * problem_in.Lz;
    Differential(workspace, T_in, Q_in, problem_in.amp, problem_in.r0, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);
    for (unsigned i = 0u; i < L; i++)
    {
        T_out[i] = T_in[i] + problem_in.dt * (workspace[i]);
    }
}

void HeatConductionRadiationCylinder::CPU::AllocWorkspaceRK4(double *&workspace_out, unsigned length_in)
{
    workspace_out = new double[5u * length_in];
}

void HeatConductionRadiationCylinder::CPU::FreeWorkspaceRK4(double *&workspace_in)
{
    delete[] workspace_in;
    workspace_in = NULL;
}

void HeatConductionRadiationCylinder::CPU::RK4(double *T_out, const double *T_in, const double *Q_in, HeatConductionRadiationCylinderProblem &problem_in, double *workspace)
{
    unsigned L = problem_in.Lr * problem_in.Lth * problem_in.Lz;
    double *k1, *k2, *k3, *k4;
    double *T_aux;
    T_aux = workspace;
    k1 = T_aux + L;
    k2 = k1 + L;
    k3 = k2 + L;
    k4 = k3 + L;
    Differential(k1, T_in, Q_in, problem_in.amp, problem_in.r0, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i] + 0.5 * problem_in.dt * k1[i];
    }
    Differential(k2, T_aux, Q_in, problem_in.amp, problem_in.r0, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i] + 0.5 * problem_in.dt * k2[i];
    }
    Differential(k3, T_aux, Q_in, problem_in.amp, problem_in.r0, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_aux[i] = T_in[i] + problem_in.dt * k3[i];
    }
    Differential(k4, T_aux, Q_in, problem_in.amp, problem_in.r0, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz);

    for (unsigned i = 0u; i < L; i++)
    {
        T_out[i] = T_in[i] + (problem_in.dt / 6.0) * (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]);
    }
}

void HeatConductionRadiationCylinder::CPU::SetFlux(double *Q_out, HeatConductionRadiationCylinderProblem &problem_in, unsigned t_in)
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

void HeatConductionRadiationCylinder::CPU::AddError(double *T_out, double mean_in, double sigma_in, unsigned length)
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
    return 0.5;
}

__device__ inline double DifferentialK(double TN_in, double T_in, double TP_in, double delta_in, double coefN, double coefP)
{
    double auxN = 2.0 * (K(TN_in) * K(T_in)) / (K(TN_in) + K(T_in)) * (TN_in - T_in) / delta_in;
    double auxP = 2.0 * (K(TP_in) * K(T_in)) / (K(TP_in) + K(T_in)) * (TP_in - T_in) / delta_in;
    return (coefN * auxN + coefP * auxP) / delta_in;
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
    diff_aux += DifferentialK(T[thread - 1u], T[thread], T[thread + 1u], dr, 1.0, 1.0);
    diff_aux += DifferentialK(T[thread - threadDimX], T[thread], T[thread + threadDimX], dth, 1.0 / (r0 - (dr * Lr) / 2.0), 1.0 / (r0 + (dr * Lr) / 2.0));
    diff_aux += DifferentialK(T[thread - threadDimX * threadDimY], T[thread], T[thread + threadDimX * threadDimY], dz, 1.0, 1.0);

    if (inside)
    {
        diff_out[index] = diff_aux;
    }
}

__global__ void TermalCapacity(double *diff_out, const double *T_in, unsigned Lr, unsigned Lth, unsigned Lz)
{
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned index = Index3D(xIndex, yIndex, zIndex, Lr, Lth, Lz);
    if (xIndex < Lr && yIndex < Lth && zIndex < Lz)
    {
        diff_out[index] /= C(T_in[index]);
    }
}

void HeatConductionRadiationCylinder::GPU::Differential(double *diff_out, const double *T_in, const double *Q_in, double amp, double r0, double dr, double dth, double dz, unsigned Lr, unsigned Lth, unsigned Lz, cublasHandle_t handle_in, cudaStream_t stream_in)
{
    cublasSetStream(handle_in, stream_in);
    // Temperature Diffusion
    dim3 T(8u, 8u, 4u);
    dim3 B((Lr + T.x - 1u) / T.x, (Lth + T.y - 1u) / T.y, (Lz + T.z - 1u) / T.z);
    unsigned size = sizeof(double) * (T.x + 2u) * (T.y + 2u) * (T.z + 2u);
    DifferentialAxis<<<B, T, size, stream_in>>>(diff_out, T_in, r0, dr, dth, dz, Lr, Lth, Lz);
    // Flux Contribution
    double alpha = amp / dz;
    cublasDaxpy(handle_in, Lr * Lth, &alpha, Q_in, 1, diff_out + Lr * Lth * (Lz - 1u), 1);
    // Thermal Capacity
    TermalCapacity<<<B, T, size, stream_in>>>(diff_out, T_in, Lr, Lth, Lz);
}

void HeatConductionRadiationCylinder::GPU::AllocWorkspaceEuler(double *&workspace_out, unsigned length_in, cudaStream_t stream_in)
{
    cudaMallocAsync(&workspace_out, sizeof(double) * length_in, stream_in);
}

void HeatConductionRadiationCylinder::GPU::FreeWorkspaceEuler(double *&workspace_out, cudaStream_t stream_in)
{
    cudaFreeAsync(workspace_out, stream_in);
}

void HeatConductionRadiationCylinder::GPU::Euler(double *T_out, double *T_in, double *Q_in, HeatConductionRadiationCylinderProblem &problem_in, double *workspace, cublasHandle_t handle_in, cudaStream_t stream_in)
{
    cublasDcopy(handle_in, problem_in.Lr * problem_in.Lth * problem_in.Lz, T_in, 1u, T_out, 1u);
    Differential(workspace, T_in, Q_in, problem_in.amp, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);
    double alpha = problem_in.dt;
    cublasDaxpy(handle_in, problem_in.Lr * problem_in.Lth * problem_in.Lz, &alpha, workspace, 1u, T_out, 1u);
}
void HeatConductionRadiationCylinder::GPU::AllocWorkspaceRK4(double *&workspace_out, unsigned length_in, cudaStream_t stream_in)
{
    cudaMallocAsync(&workspace_out, sizeof(double) * 5u * length_in, stream_in);
}
void HeatConductionRadiationCylinder::GPU::FreeWorkspaceRK4(double *&workspace_out, cudaStream_t stream_in)
{
    cudaFreeAsync(workspace_out, stream_in);
}
void HeatConductionRadiationCylinder::GPU::RK4(double *T_out, double *T_in, double *Q_in, HeatConductionRadiationCylinderProblem &problem_in, double *workspace, cublasHandle_t handle_in, cudaStream_t stream_in)
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
    Differential(k1, aux, Q_in, problem_in.amp, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

    cublasDcopy(handle_in, L, T_in, 1u, aux, 1u);
    alpha = 0.5 * problem_in.dt;
    cublasDaxpy(handle_in, L, &alpha, k1, 1u, aux, 1u);
    Differential(k2, aux, Q_in, problem_in.amp, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

    cublasDcopy(handle_in, L, T_in, 1u, aux, 1u);
    alpha = 0.5 * problem_in.dt;
    cublasDaxpy(handle_in, L, &alpha, k2, 1u, aux, 1u);
    Differential(k3, aux, Q_in, problem_in.amp, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

    cublasDcopy(handle_in, L, T_in, 1u, aux, 1u);
    alpha = 1.0 * problem_in.dt;
    cublasDaxpy(handle_in, L, &alpha, k3, 1u, aux, 1u);
    Differential(k4, aux, Q_in, problem_in.amp, problem_in.dr, problem_in.dth, problem_in.dz, problem_in.Lr, problem_in.Lth, problem_in.Lz, handle_in, stream_in);

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

void HeatConductionRadiationCylinder::GPU::SetFlux(double *Q_out, HeatConductionRadiationCylinderProblem &problem_in, unsigned t_in, cudaStream_t stream_in)
{
    dim3 T(16u, 16u);
    dim3 B((problem_in.Lr + T.x - 1u) / T.x, (problem_in.Lth + T.y - 1u) / T.y);
    SetFluxDevice<<<B, T, 0, stream_in>>>(Q_out, problem_in.dr, problem_in.dth, problem_in.Sr, problem_in.Sth, problem_in.Lr, problem_in.Lth);
}

void HeatConductionRadiationCylinder::GPU::AddError(double *T_out, double mean_in, double sigma_in, unsigned length, cudaStream_t stream_in)
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