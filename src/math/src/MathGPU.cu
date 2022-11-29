#include "../include/MathGPU.hpp"

#define THREAD_COUNT 1024u

// Auxiliary Functions
void MathGPU::Print(Pointer<double> vector_in, unsigned length_in, cudaStream_t stream_in)
{
    Pointer<double> vector_aux = MemoryHandler::Alloc<double>(length_in, PointerType::CPU, PointerContext::GPU_Aware, stream_in);
    MemoryHandler::Copy(vector_aux, vector_in, length_in, stream_in);
    cudaStreamSynchronize(stream_in);
    std::cout << "Vector -> Size : " << length_in << "\n";
    for (unsigned i = 0u; i < length_in; i++)
    {
        std::cout << " " << vector_aux.pointer[i] << "\n";
    }
    MemoryHandler::Free<double>(vector_aux, stream_in);
}

void MathGPU::Print(Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, cudaStream_t stream_in)
{
    Pointer<double> matrix_aux = MemoryHandler::Alloc<double>(lengthX_in * lengthY_in, PointerType::CPU, PointerContext::GPU_Aware, stream_in);
    MemoryHandler::Copy(matrix_aux, matrix_in, lengthX_in * lengthY_in, stream_in);
    cudaStreamSynchronize(stream_in);
    std::cout << "Matrix -> Size : " << lengthX_in << ":" << lengthY_in << "\n";
    std::cout << std::scientific;
    for (unsigned i = 0u; i < lengthX_in; i++)
    {
        for (unsigned j = 0u; j < lengthY_in; j++)
        {
            std::cout << " " << matrix_aux.pointer[j * lengthX_in + i];
        }
        std::cout << "\n";
    }
    MemoryHandler::Free<double>(matrix_aux, stream_in);
}

// In-Placed Calculation

// Device Functions

__global__ void CUDA_Add(double *vector_inout, double *vector_in, unsigned length_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length_in)
    {
        vector_inout[index] += vector_in[index];
    }
}

__global__ void CUDA_Sub(double *vector_inout, double *vector_in, unsigned length_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length_in)
    {
        vector_inout[index] -= vector_in[index];
    }
}

__global__ void CUDA_Mul(double *vector_inout, double value_in, unsigned length_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length_in)
    {
        vector_inout[index] *= value_in;
    }
}

__global__ void CUDA_Mul(double *vector_inout, double *vector_in, unsigned length_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length_in)
    {
        vector_inout[index] *= vector_in[index];
    }
}

// Vector Element-wise Addition
void MathGPU::Add(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in, cudaStream_t stream_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Add<<<B, T, 0u, stream_in>>>(vector_inout.pointer, vector_in.pointer, length_in);
}
// Vector Element-wise Subtraction
void MathGPU::Sub(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in, cudaStream_t stream_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Sub<<<B, T, 0u, stream_in>>>(vector_inout.pointer, vector_in.pointer, length_in);
}
// Vector Constant Multiplication
void MathGPU::Mul(Pointer<double> vector_inout, double value_in, unsigned length_in, cudaStream_t stream_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Mul<<<B, T, 0u, stream_in>>>(vector_inout.pointer, value_in, length_in);
}
// Vector Element-wise Multiplication
void MathGPU::Mul(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in, cudaStream_t stream_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Mul<<<B, T, 0u, stream_in>>>(vector_inout.pointer, vector_in.pointer, length_in);
}

// Out-Placed Calculation

// Device Functions

__global__ void CUDA_Add(double *vector_out, double *vectorLeft_in, double *vectorRight_in, unsigned length_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length_in)
    {
        vector_out[index] = vectorLeft_in[index] + vectorRight_in[index];
    }
}

__global__ void CUDA_Sub(double *vector_out, double *vectorLeft_in, double *vectorRight_in, unsigned length_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length_in)
    {
        vector_out[index] = vectorLeft_in[index] - vectorRight_in[index];
    }
}

__global__ void CUDA_Mul(double *vector_out, double *vectorLeft_in, double valueRight_in, unsigned length_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length_in)
    {
        vector_out[index] = vectorLeft_in[index] * valueRight_in;
    }
}

__global__ void CUDA_Mul(double *vector_out, double *vectorLeft_in, double *vectorRight_in, unsigned length_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length_in)
    {
        vector_out[index] = vectorLeft_in[index] * vectorRight_in[index];
    }
}

// Vector Element-wise Addition
void MathGPU::Add(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, cudaStream_t stream_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Add<<<B, T, 0u, stream_in>>>(vector_out.pointer, vectorLeft_in.pointer, vectorRight_in.pointer, length_in);
}
// Vector Element-wise Subtraction
void MathGPU::Sub(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, cudaStream_t stream_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Sub<<<B, T, 0u, stream_in>>>(vector_out.pointer, vectorLeft_in.pointer, vectorRight_in.pointer, length_in);
}
// Vector Constant Multiplication
void MathGPU::Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in, cudaStream_t stream_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Mul<<<B, T, 0u, stream_in>>>(vector_out.pointer, vectorLeft_in.pointer, valueRight_in, length_in);
}
// Vector Element-wise Multiplication
void MathGPU::Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, cudaStream_t stream_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Mul<<<B, T, 0u, stream_in>>>(vector_out.pointer, vectorLeft_in.pointer, vectorRight_in.pointer, length_in);
}

template <typename T>
void Swap(T &left_inout, T &right_inout)
{
    T aux = left_inout;
    left_inout = right_inout;
    right_inout = aux;
}

void SetCuBLASOperation(double *&auxLeft_out, cublasOperation_t &leftStructure_out, unsigned &lengthXLeft_out, unsigned &lengthYLeft_out, unsigned &leadingDimensionLeft_out,
                        double *&auxRight_out, cublasOperation_t &rightStructure_out, unsigned &lengthXRight_out, unsigned &lengthYRight_out, unsigned &leadingDimensionRight_out,
                        unsigned &lengthXOut_in, unsigned &lengthYOut_in,
                        MatrixStructure matrixOutStructure_in, MatrixStructure matrixLeftStructure_in, MatrixStructure matrixRightStructure_in)
{
    leftStructure_out = cublasOperation_t::CUBLAS_OP_N;
    rightStructure_out = cublasOperation_t::CUBLAS_OP_N;
    if (matrixLeftStructure_in == MatrixStructure_Transposed)
    {
        leftStructure_out = cublasOperation_t::CUBLAS_OP_T;
        Swap(lengthXLeft_out, lengthYLeft_out);
    }
    if (matrixRightStructure_in == MatrixStructure_Transposed)
    {
        rightStructure_out = cublasOperation_t::CUBLAS_OP_T;
        Swap(lengthXRight_out, lengthYRight_out);
    }
    if (matrixOutStructure_in == MatrixStructure_Transposed)
    {
        Swap(lengthXOut_in, lengthYOut_in);
        Swap(auxLeft_out, auxRight_out);
        Swap(lengthXLeft_out, lengthYRight_out);
        Swap(lengthYLeft_out, lengthXRight_out);
        Swap(leadingDimensionLeft_out, leadingDimensionRight_out);
        leftStructure_out = (matrixRightStructure_in == MatrixStructure_Transposed) ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T;
        rightStructure_out = (matrixLeftStructure_in == MatrixStructure_Transposed) ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T;
    }
}

// Matrix Multiplication
void MathGPU::MatrixMultiplication(double alpha,
                                   Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                                   Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
                                   double beta,
                                   Pointer<double> matrix_out, MatrixStructure matrixOutStructure_in, unsigned lengthOutX_in, unsigned lengthOutY_in,
                                   cublasHandle_t handle_in, cudaStream_t stream_in, Pointer<double> weight_in)
{
    double *aux = NULL;
    double *auxL, *auxR;
    unsigned ldL, ldR;
    cublasOperation_t opL, opR;
    unsigned ML, MX;
    unsigned NR, NX;
    unsigned KL, KR;
    unsigned K;

    ldL = lengthLeftX_in;
    ldR = lengthRightX_in;

    ML = lengthLeftX_in;
    MX = lengthOutX_in;
    NR = lengthRightY_in;
    NX = lengthOutY_in;
    KL = lengthLeftY_in;
    KR = lengthRightX_in;
    cublasSetStream(handle_in, stream_in);
    auxL = matrixLeft_in.pointer;
    auxR = matrixRight_in.pointer;

    SetCuBLASOperation(auxL, opL, ML, KL, ldL, auxR, opR, KR, NR, ldR, MX, NX, matrixOutStructure_in, matrixLeftStructure_in, matrixRightStructure_in);

    if (KL == KR && MX == ML && NX == NR)
    {
        K = KL;
    }
    else
    {
        std::cout << "Error: Sizes do not match.\n";
        std::cout << "M: " << ML << " " << MX << "\n";
        std::cout << "N: " << NR << " " << NX << "\n";
        std::cout << "K: " << KL << " " << KR << "\n";
        return;
    }
    if (weight_in.pointer != NULL)
    {
        if (ML < NR)
        {
            cudaMallocAsync(&aux, sizeof(double) * ML * KL, stream_in);
            for (unsigned i = 0u; i < K; i++)
            {
                unsigned stride1, stride2;
                stride1 = (opL == CUBLAS_OP_N) ? 1 : ldL;
                stride2 = (opL == CUBLAS_OP_N) ? ldL : 1;
                cublasDaxpy(handle_in, ML, weight_in.pointer + i, auxL + i * stride2, stride1, aux + i * stride2, stride1);
            }
            cublasDgemm(handle_in, opL, opR, MX, NX, K, &alpha, aux, ldL, auxR, ldR, &beta, matrix_out.pointer, MX);
            cudaFreeAsync(aux, stream_in);
        }
        else
        {
            cudaMallocAsync(&aux, sizeof(double) * KR * NR, stream_in);
            for (unsigned i = 0u; i < K; i++)
            {
                unsigned stride1, stride2;
                stride1 = (opL == CUBLAS_OP_N) ? ldR : 1;
                stride2 = (opL == CUBLAS_OP_N) ? 1 : ldR;
                cublasDaxpy(handle_in, NR, weight_in.pointer + i, auxR + i * stride2, stride1, aux + i, stride1);
            }
            cublasDgemm(handle_in, opL, opR, MX, NX, K, &alpha, auxL, ldL, aux, ldR, &beta, matrix_out.pointer, MX);
            cudaFreeAsync(aux, stream_in);
        }
    }
    else
    {
        cublasDgemm(handle_in, opL, opR, MX, NX, K, &alpha, auxL, ldL, auxR, ldR, &beta, matrix_out.pointer, MX);
    }
}

// Reducibles Operations
void MathGPU::Mean(Pointer<double> vector_out, Pointer<double> matrixLeft_in, unsigned lengthX_in, unsigned lengthY_in, cublasHandle_t handle_in, cudaStream_t stream_in, Pointer<double> weight_in)
{
    double alpha, beta;
    cublasSetStream(handle_in, stream_in);
    if (weight_in.pointer != NULL)
    {
        alpha = 1.0;
        beta = 1.0;
        cublasDgemv(handle_in, CUBLAS_OP_N, lengthX_in, lengthY_in, &alpha, matrixLeft_in.pointer, lengthX_in, weight_in.pointer, 1, &beta, vector_out.pointer, 1);
    }
    else
    {
        alpha = 1.0 / lengthY_in;
        for (unsigned i = 0u; i < lengthY_in; i++)
        {
            cublasDaxpy(handle_in, lengthX_in, &alpha, matrixLeft_in.pointer, 1, vector_out.pointer, 1);
        }
    }
}

bool MathGPU::Compare(Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, cudaStream_t stream_in)
{
    Pointer<double> p1_aux, p2_aux;
    p1_aux = MemoryHandler::Alloc<double>(length_in, PointerType::CPU, PointerContext::GPU_Aware, stream_in);
    p2_aux = MemoryHandler::Alloc<double>(length_in, PointerType::CPU, PointerContext::GPU_Aware, stream_in);
    MemoryHandler::Copy<double>(p1_aux, vectorLeft_in, length_in, stream_in);
    MemoryHandler::Copy<double>(p2_aux, vectorRight_in, length_in, stream_in);

    cudaStreamSynchronize(stream_in);

    bool out = true;
    for (unsigned i = 0u; i < length_in; i++)
    {
        if (p1_aux.pointer[i] != p2_aux.pointer[i])
        {
            out = false;
            break;
        }
    }
    MemoryHandler::Free<double>(p1_aux, stream_in);
    MemoryHandler::Free<double>(p2_aux, stream_in);
    return out;
}

// Linear System Solvers

// Helper Functions

// Wrapper Methods
void MathGPU::Decomposition(Pointer<double> decomposition_out, DecompositionType decompositionType_in, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, cusolverDnHandle_t handle_in, cudaStream_t stream_in, Pointer<double> pivot_out)
{
    size_t sizeDevice = 0u;
    size_t sizeHost = 0u;
    double *workspaceDevice = NULL;
    double *workspaceHost = NULL;
    int *infoDevice = NULL;
    int *infoHost = new int(0);
    cusolverDnParams_t params = NULL;
    cusolverDnSetStream(handle_in, stream_in);
    cudaMallocAsync(&infoDevice, sizeof(int), stream_in);
    switch (decompositionType_in)
    {
    case DecompositionType_Cholesky:
        MemoryHandler::Copy(decomposition_out, matrix_in, lengthX_in * lengthY_in, stream_in);
        cusolverDnXpotrf_bufferSize(handle_in, params, CUBLAS_FILL_MODE_LOWER, lengthX_in, CUDA_R_64F, matrix_in.pointer, lengthX_in, CUDA_R_64F, &sizeDevice, &sizeHost);
        cudaStreamSynchronize(stream_in);
        if (sizeDevice > 0)
        {
            cudaMallocAsync(&workspaceDevice, sizeDevice, stream_in);
        }
        if (sizeHost > 0)
        {
            cudaMallocHost(&workspaceHost, sizeHost);
        }
        cusolverDnXpotrf(handle_in, params, CUBLAS_FILL_MODE_LOWER, lengthX_in, CUDA_R_64F, decomposition_out.pointer, lengthX_in, CUDA_R_64F, workspaceDevice, sizeDevice, workspaceHost, sizeHost, infoDevice);
        cudaMemcpyAsync(infoHost, infoDevice, sizeof(int), cudaMemcpyDeviceToHost, stream_in);
        if (sizeDevice > 0)
        {
            cudaFreeAsync(workspaceDevice, stream_in);
        }
        if (sizeHost > 0)
        {
            cudaFreeHost(workspaceHost);
        }
        cudaStreamSynchronize(stream_in);
        if (*infoHost != 0)
        {
            std::cout << "Info : " << *infoHost << "\n";
        }
        break;
    default:
        break;
    }

    cudaFreeAsync(infoDevice, stream_in);
    delete infoHost;
}

void MathGPU::Solve(Pointer<double> X_out, LinearSolverType solverType_in,
                    Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
                    Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in,
                    cusolverDnHandle_t solverHandle_in, cudaStream_t stream_in, cublasHandle_t cublasHandle_in)
{
    int *infoDevice = NULL;
    int *infoHost = new int(0);
    Pointer<double> decomposition = MemoryHandler::Alloc<double>(lengthAX_in * lengthAY_in, PointerType::GPU, PointerContext::GPU_Aware, stream_in);
    cusolverDnParams_t params = NULL;
    cusolverDnSetStream(solverHandle_in, stream_in);
    cublasSetStream(cublasHandle_in, stream_in);
    switch (solverType_in)
    {
    case LinearSolverType_Cholesky:
        MemoryHandler::Copy(X_out, B_in, lengthAY_in * lengthBY_in, stream_in);
        MathGPU::Decomposition(decomposition, DecompositionType_Cholesky, A_in, lengthAX_in, lengthAY_in, solverHandle_in, stream_in);
        MathGPU::Print(decomposition, lengthAX_in, lengthAY_in, stream_in);
        cusolverDnXpotrs(solverHandle_in, params, CUBLAS_FILL_MODE_LOWER, lengthAX_in, lengthBY_in, CUDA_R_64F, decomposition.pointer, lengthAX_in, CUDA_R_64F, X_out.pointer, lengthBX_in, infoDevice);
        cudaMemcpyAsync(infoHost, infoDevice, sizeof(int), cudaMemcpyDeviceToHost, stream_in);
        cudaStreamSynchronize(stream_in);
        if (*infoHost != 0)
        {
            std::cout << "Info : " << *infoHost << "\n";
        }
        break;
    default:
        break;
    }
    MemoryHandler::Free<double>(decomposition, stream_in);
    cudaFreeAsync(infoDevice, stream_in);
    delete infoHost;
}
