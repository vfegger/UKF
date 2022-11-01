#include "../include/MathGPU.hpp"

#define THREAD_COUNT 1024u

// Auxiliary Functions
void MathGPU::Print(Pointer<double> vector_in, unsigned length_in)
{
    Pointer<double> vector_aux = MemoryHandler::Alloc<double>(length_in, PointerType::CPU, PointerContext::GPU_Aware);
    MemoryHandler::Copy(vector_aux, vector_in, length_in);
    cudaDeviceSynchronize();
    std::cout << "Vector -> Size : " << length_in << "\n";
    for (unsigned i = 0u; i < length_in; i++)
    {
        std::cout << " " << vector_aux.pointer[i] << "\n";
    }
    MemoryHandler::Free<double>(vector_aux);
}

void MathGPU::Print(Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in)
{
    Pointer<double> matrix_aux = MemoryHandler::Alloc<double>(lengthX_in * lengthY_in, PointerType::CPU, PointerContext::GPU_Aware);
    MemoryHandler::Copy(matrix_aux, matrix_in, lengthX_in * lengthY_in);
    cudaDeviceSynchronize();
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
    MemoryHandler::Free<double>(matrix_aux);
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
void MathGPU::Add(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Add<<<B, T>>>(vector_inout.pointer, vector_in.pointer, length_in);
}
// Vector Element-wise Subtraction
void MathGPU::Sub(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Sub<<<B, T>>>(vector_inout.pointer, vector_in.pointer, length_in);
}
// Vector Constant Multiplication
void MathGPU::Mul(Pointer<double> vector_inout, double value_in, unsigned length_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Mul<<<B, T>>>(vector_inout.pointer, value_in, length_in);
}
// Vector Element-wise Multiplication
void MathGPU::Mul(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Mul<<<B, T>>>(vector_inout.pointer, vector_in.pointer, length_in);
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
void MathGPU::Add(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Add<<<B, T>>>(vector_out.pointer, vectorLeft_in.pointer, vectorRight_in.pointer, length_in);
}
// Vector Element-wise Subtraction
void MathGPU::Sub(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Sub<<<B, T>>>(vector_out.pointer, vectorLeft_in.pointer, vectorRight_in.pointer, length_in);
}
// Vector Constant Multiplication
void MathGPU::Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Mul<<<B, T>>>(vector_out.pointer, vectorLeft_in.pointer, valueRight_in, length_in);
}
// Vector Element-wise Multiplication
void MathGPU::Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    dim3 T(THREAD_COUNT);
    dim3 B((length_in + T.x - 1u) / T.x);
    CUDA_Mul<<<B, T>>>(vector_out.pointer, vectorLeft_in.pointer, vectorRight_in.pointer, length_in);
}
// Matrix Multiplication TODO
void MathGPU::MatrixMultiplication(double alpha,
                                   Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                                   Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
                                   double beta,
                                   Pointer<double> matrix_out, MatrixStructure matrixOutStructure_in, unsigned lengthOutX_in, unsigned lengthOutY_in,
                                   Pointer<double> weight_in)
{
    unsigned m, n, k;
    unsigned lda, ldb, ldc;
    cublasHandle_t handle;
    m = lengthLeftX_in;
    n = lengthRightY_in;
    k = lengthLeftY_in;
    cublasDgemm_v2(handle,
                   cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N,
                   m, n, k,
                   &alpha, matrixLeft_in.pointer, lda, matrixRight_in.pointer, ldb,
                   &beta, matrix_out.pointer, ldc);
}

// Reducibles Operations
void MathGPU::Mean(Pointer<double> vector_out, Pointer<double> matrixLeft_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> weight_in)
{
    double alpha, beta;
    cublasHandle_t handle;
    if (weight_in.pointer != NULL)
    {
        alpha = 1.0;
        beta = 1.0;
        cublasDgemv_v2(handle, CUBLAS_OP_N, lengthX_in, lengthY_in, &alpha, matrixLeft_in.pointer, lengthX_in, weight_in.pointer, 1, &beta, vector_out.pointer, 1);
    }
    else
    {
        alpha = 1.0 / lengthY_in;
        for (unsigned i = 0u; i < lengthY_in; i++)
        {
            cublasDaxpy_v2(handle, lengthX_in, &alpha, matrixLeft_in.pointer, 1, vector_out.pointer, 1);
        }
    }
}

bool MathGPU::Compare(Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    Pointer<double> p1_aux, p2_aux;
    p1_aux = MemoryHandler::Alloc<double>(length_in, PointerType::CPU, PointerContext::GPU_Aware);
    p2_aux = MemoryHandler::Alloc<double>(length_in, PointerType::CPU, PointerContext::GPU_Aware);
    MemoryHandler::Copy<double>(p1_aux, vectorLeft_in, length_in);
    MemoryHandler::Copy<double>(p2_aux, vectorRight_in, length_in);

    cudaDeviceSynchronize();

    bool out = true;
    for (unsigned i = 0u; i < length_in; i++)
    {
        if (p1_aux.pointer[i] != p2_aux.pointer[i])
        {
            out = false;
            break;
        }
    }
    MemoryHandler::Free<double>(p1_aux);
    MemoryHandler::Free<double>(p2_aux);
    return out;
}

// Linear System Solvers

// Helper Functions

// Wrapper Methods
void MathGPU::Decomposition(Pointer<double> decomposition_out, DecompositionType decompositionType_in, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> pivot_out)
{
    size_t size = 0u;
    double *workspace = NULL;
    int info = 0;
    cusolverDnHandle_t handle;
    cusolverDnParams_t params;
    switch (decompositionType_in)
    {
    case DecompositionType_Cholesky:
        MemoryHandler::Copy(decomposition_out, matrix_in, lengthX_in * lengthY_in);
        cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, lengthX_in, CUDA_R_64F, matrix_in.pointer, lengthX_in, CUDA_R_64F, &size);
        cudaMalloc(&workspace, size);
        cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, lengthX_in, CUDA_R_64F, decomposition_out.pointer, lengthX_in, CUDA_R_64F, workspace, size, &info);
        break;
    default:
        break;
    }
}

void MathGPU::Solve(Pointer<double> X_out, LinearSolverType solverType_in, MatrixOperationSide operationSide_in,
                    Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
                    Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in)
{
    int info = 0;
    Pointer<double> decomposition = MemoryHandler::Alloc<double>(lengthAX_in * lengthAY_in, PointerType::GPU, PointerContext::GPU_Aware);
    cusolverDnHandle_t handle;
    cusolverDnParams_t params;
    switch (solverType_in)
    {
    case LinearSolverType_Cholesky:
        MathGPU::Decomposition(decomposition, DecompositionType_Cholesky, A_in, lengthAX_in, lengthAY_in);
        MemoryHandler::Copy(X_out, B_in, lengthBX_in * lengthBY_in);
        cusolverDnPotrs(handle, params, CUBLAS_FILL_MODE_LOWER, lengthAX_in, lengthBY_in, CUDA_R_64F, decomposition.pointer, lengthAX_in, CUDA_R_64F, X_out.pointer, lengthBX_in, &info);
        break;
    default:
        break;
    }
    MemoryHandler::Free<double>(decomposition);
}
