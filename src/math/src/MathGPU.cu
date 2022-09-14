#include "../include/MathGPU.hpp"

void MathGPU::InitializeGPUContext(int device)
{
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess)
    {
        cudaGetDeviceProperties(MathGPU::properties, device);
    }
    else
    {
        std::cout << "Cuda-enabled context was not able to launch properly with the chosen device.\n";
    }
}

void MathGPU::PrintMatrix(double *matrix_in, unsigned lengthX_in, unsigned lengthY_in, unsigned precision)
{
    // Get values from GPU pointer to CPU pointer
    double *matrix_aux = new double[lengthX_in * lengthY_in];
    cudaMemcpy(matrix_aux, matrix_in, sizeof(double) * lengthX_in * lengthY_in, cudaMemcpyDeviceToHost);

    std::cout << "Sizes: X = " << lengthX_in << "; Y = " << lengthY_in << "\n";
    std::cout.precision(precision);
    std::cout << std::scientific;
    for (unsigned i = 0u; i < lengthX_in; i++)
    {
        std::cout << "| ";
        for (unsigned j = 0u; j < lengthY_in; j++)
        {
            std::cout << matrix_aux[j * lengthX_in + i];
        }
        std::cout << " |\n";
    }
}

// Out-Placed Calculation Device

__device__ void _Add(double *vector_out, double *vectorLeft_in, double *vectorRight_in, unsigned length_in)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length_in)
    {
        vector_out[index] = vectorLeft_in[index] + vectorRight_in[index];
    }
}
__device__ void _Sub(double *vector_out, double *vectorLeft_in, double *vectorRight_in, unsigned length_in)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length_in)
    {
        vector_out[index] = vectorLeft_in[index] - vectorRight_in[index];
    }
}
__device__ void _Mul(double *vector_out, double *vectorLeft_in, double *vectorRight_in, unsigned length_in)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length_in)
    {
        vector_out[index] = vectorLeft_in[index] * vectorRight_in[index];
    }
}
__device__ void _Mul(double *vector_out, double *vectorLeft_in, double value_in, unsigned length_in)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length_in)
    {
        vector_out[index] = vectorLeft_in[index] * value_in;
    }
}

// Out-Placed Calculation Host
void MathGPU::Add(double *vector_out, double *vectorLeft_in, double *vectorRight_in, unsigned length_in, cudaStream_t stream)
{
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in + T.x - 1u) / T.x);
    _Add<<<T, B, 0, stream>>>(vector_out, vectorLeft_in, vectorRight_in, length_in);
}
void MathGPU::Sub(double *vector_out, double *vectorLeft_in, double *vectorRight_in, unsigned length_in, cudaStream_t stream)
{
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in + T.x - 1u) / T.x);
    _Sub<<<T, B, 0, stream>>>(vector_out, vectorLeft_in, vectorRight_in, length_in);
}
void MathGPU::Mul(double *vector_out, double *vectorLeft_in, double *vectorRight_in, unsigned length_in, cudaStream_t stream)
{
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in + T.x - 1u) / T.x);
    _Mul<<<T, B, 0, stream>>>(vector_out, vectorLeft_in, vectorRight_in, length_in);
}
void MathGPU::Mul(double *vector_out, double *vectorLeft_in, double value_in, unsigned length_in, cudaStream_t stream)
{
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in + T.x - 1u) / T.x);
    _Mul<<<T, B, 0, stream>>>(vector_out, vectorLeft_in, value_in, length_in);
}

// Out-Placed Calculation Device
__device__ void _Add(double *vector_inout, double *vector_in, unsigned length_in)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length_in)
    {
        vector_inout[index] += vector_in[index];
    }
}
__device__ void _Sub(double *vector_inout, double *vector_in, unsigned length_in)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length_in)
    {
        vector_inout[index] -= vector_in[index];
    }
}
__device__ void _Mul(double *vector_inout, double *vector_in, unsigned length_in)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length_in)
    {
        vector_inout[index] *= vector_in[index];
    }
}
__device__ void _Mul(double *vector_inout, double value_in, unsigned length_in)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length_in)
    {
        vector_inout[index] *= value_in;
    }
}

// In-Placed Calculation
void MathGPU::Add(double *vector_inout, double *vector_in, unsigned length_in, cudaStream_t stream)
{
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in + T.x - 1u) / T.x);
    _Add<<<T, B, 0, stream>>>(vector_inout, vector_in, length_in);
}
void MathGPU::Sub(double *vector_inout, double *vector_in, unsigned length_in, cudaStream_t stream)
{
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in + T.x - 1u) / T.x);
    _Sub<<<T, B, 0, stream>>>(vector_inout, vector_in, length_in);
}
void MathGPU::Mul(double *vector_inout, double *vector_in, unsigned length_in, cudaStream_t stream)
{
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in + T.x - 1u) / T.x);
    _Mul<<<T, B, 0, stream>>>(vector_inout, vector_in, length_in);
}
void MathGPU::Mul(double *vector_inout, double value_in, unsigned length_in, cudaStream_t stream)
{
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in + T.x - 1u) / T.x);
    _Mul<<<T, B, 0, stream>>>(vector_inout, value_in, length_in);
}

cublasOperation_t MapStructure(MatrixStructure structure)
{
    cublasOperation_t operation;
    switch (structure)
    {
    case MatrixStructure::Natural:
        operation = cublasOperation_t::CUBLAS_OP_N;
        break;
    case MatrixStructure::Transposed:
        operation = cublasOperation_t::CUBLAS_OP_T;
        break;
    default:
        std::cout << "Structure not defined for the left matrix. Using default structure.\n";
        return CUBLAS_OP_N;
        break;
    }
    return operation;
}

// Matrix Multiplication
void MathGPU::MatrixMultiplication(double *matrix_out, double alpha, double beta,
                                   double *matrixLeft_in, MatrixStructure structureLeft, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                                   double *matrixRight_in, MatrixStructure structureRight, unsigned lengthRightX_in, unsigned lengthRightY_in,
                                   double *weight_in, cudaStream_t stream)
{
    cublasHandle_t handle;
    cublasOperation_t left_OP, right_OP;
    cublasCreate_v2(&handle);
    cublasSetStream_v2(handle, stream);
    cublasOperation_t left_OP = MapStructure(structureLeft);
    cublasOperation_t right_OP = MapStructure(structureRight);
    if (lengthLeftY_in != lengthRightX_in)
    {
        std::cout << "Error: multiplication sizes do not match.\n";
        return;
    }
    cublasDgemm_v2(handle, left_OP, right_OP, lengthLeftX_in, lengthRightY_in, lengthLeftY_in, &alpha, matrixLeft_in, lengthLeftX_in, matrixRight_in, lengthRightX_in, &beta, matrix_out, lengthLeftX_in);
    cublasDestroy_v2(handle);
}

// Operators
void MathGPU::Operation(void (*operation_in)(double *matrix_out, double *matrixLeft_in, double *matrixRight_in, unsigned length_in),
                        double *matrix_out, double *matrixLeft_in, double *matrixRight_in, unsigned length_in,
                        unsigned strideOutput_in, unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in, unsigned offsetOutput_in, unsigned offsetLeft_in, unsigned offsetRight_in)
{
    double *output = matrix_out + offsetOutput_in;
    double *left = matrixLeft_in + offsetLeft_in;
    double *right = matrixRight_in + offsetRight_in;
    if (length_in < strideLeft_in)
    {
        std::cout << "Warning: The behavior is undefined to output strides lower than the length of the operation.\n";
    }
    for (unsigned i = 0u; i < iteration_in; i++)
    {
        operation_in(output + i * strideOutput_in, left + i * strideLeft_in, right + i * strideRight_in, length_in);
    }
}
void MathGPU::Operation(void (*operation_in)(double *matrixLeft_inout, double *matrixRight_in, unsigned length_in),
                        double *matrixLeft_inout, double *matrixRight_in, unsigned length_in,
                        unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in, unsigned offsetLeft_in, unsigned offsetRight_in)
{
    double *left = matrixLeft_inout + offsetLeft_in;
    double *right = matrixRight_in + offsetRight_in;
    if (length_in < strideLeft_in)
    {
        std::cout << "Warning: The behavior is undefined to left strides lower than the length of the operation.\n";
    }
    for (unsigned i = 0u; i < iteration_in; i++)
    {
        operation_in(left + i * strideLeft_in, right + i * strideRight_in, length_in);
    }
}

// Reducibles Operations
double MathGPU::Mean(double* value_out, double *vector_in, unsigned length_in, double *weight_in = NULL, bool transferResultCPU_in = true) {
    bool noWeight = weight_in == NULL;
    int stride = 1;
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    if(noWeight){
        cudaMalloc(&weight_in,sizeof(double));
        stride = 0;
        double value = 1.0/(double)length_in;
        cudaMemcpy(weight_in,&value,sizeof(double),cudaMemcpyKind::cudaMemcpyHostToDevice);
    }
    cublasDdot_v2(handle,length_in,vector_in,1,weight_in,stride,value_out);
    double res = 0.0;
    if(transferResultCPU_in) {
        cudaMemcpy(&res,value_out,1,cudaMemcpyKind::cudaMemcpyDeviceToHost);
    }
    if(noWeight){
        cudaFree(weight_in);
    }
    return res;
    
}
void Mean(double *vector_out, double *matrix_in, unsigned lengthX_in, unsigned lengthY_in, double *weight_in = NULL){
    bool noWeight = weight_in == NULL;
    int stride = 1;
    int stride = 1;
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    if(noWeight){
        cudaMalloc(&weight_in,sizeof(double));
        stride = 0;
        double value = 1.0/(double)lengthY_in;
        cudaMemcpy(weight_in,&value,sizeof(double),cudaMemcpyKind::cudaMemcpyHostToDevice);
    }
    cublasDdot_v2(handle,lengthY_in,matrix_in,lengthX_in,weight_in,stride,vector_out);
    double res = 0.0;
    if(noWeight){
        cudaFree(weight_in);
    }
    return;
}
