#include "MathGPU.hpp"

void InitializeGPUContext(int device){
    cudaError_t status = cudaSetDevice(device);
    if(status != cudaSuccess){
        cudaGetDeviceProperties(MathGPU::properties, device);
    } else {
        std::cout << "Cuda-enabled context was not able to launch properly with the chosen device.\n";
    }
}

void MathGPU::PrintMatrix(double* matrix_in, unsigned lengthX_in, unsigned lengthY_in, unsigned precision){
    // Get values from GPU pointer to CPU pointer
    double* matrix_aux = new double[lengthX_in*lengthY_in];
    cudaMemcpy(matrix_aux,matrix_in,sizeof(double)*lengthX_in*lengthY_in,cudaMemcpyDeviceToHost);

    std::cout << "Sizes: X = " << lengthX_in << "; Y = " << lengthY_in << "\n";
    std::cout.precision(precision);
    std::cout << std::scientific;
    for(unsigned i = 0u; i < lengthX_in; i++){
        std::cout << "| ";
        for(unsigned j = 0u; j < lengthY_in; j++){
            std::cout << matrix_aux[j*lengthX_in+i];
        }
        std::cout <<" |\n";
    }
}

//Out-Placed Calculation Device

__device__ void _Add(double* vector_out, double* vectorLeft_in, double* vectorRight_in, unsigned length_in){
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length_in){
        vector_out[index] = vectorLeft_in[index] + vectorRight_in[index];
    }
}
__device__ void _Sub(double* vector_out, double* vectorLeft_in, double* vectorRight_in, unsigned length_in){
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length_in){
        vector_out[index] = vectorLeft_in[index] - vectorRight_in[index];
    }
}
__device__ void _Mul(double* vector_out, double* vectorLeft_in, double* vectorRight_in, unsigned length_in){
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length_in){
        vector_out[index] = vectorLeft_in[index] * vectorRight_in[index];
    }
}
__device__ void _Mul(double* vector_out, double* vectorLeft_in, double value_in, unsigned length_in){
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length_in){
        vector_out[index] = vectorLeft_in[index] * value_in;
    }
}

//Out-Placed Calculation Host
void Add(double* vector_out, double* vectorLeft_in, double* vectorRight_in, unsigned length_in, cudaStream_t stream){
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in+T.x-1u)/T.x);
    _Add<<<T,B,0,stream>>>(vector_out, vectorLeft_in, vectorRight_in, length_in);
}
void Sub(double* vector_out, double* vectorLeft_in, double* vectorRight_in, unsigned length_in, cudaStream_t stream){
        unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in+T.x-1u)/T.x);
    _Sub<<<T,B,0,stream>>>(vector_out, vectorLeft_in, vectorRight_in, length_in);
}
void Mul(double* vector_out, double* vectorLeft_in, double* vectorRight_in, unsigned length_in, cudaStream_t stream){
        unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in+T.x-1u)/T.x);
    _Mul<<<T,B,0,stream>>>(vector_out, vectorLeft_in, vectorRight_in, length_in);
}
void Mul(double* vector_out, double* vectorLeft_in, double value_in, unsigned length_in, cudaStream_t stream){
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in+T.x-1u)/T.x);
    _Mul<<<T,B,0,stream>>>(vector_out, vectorLeft_in, value_in, length_in);
}


//Out-Placed Calculation Device
__device__ void _Add(double* vector_inout, double* vector_in, unsigned length_in){
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length_in){
        vector_inout[index] += vector_in[index];
    }
}
__device__ void _Sub(double* vector_inout, double* vector_in, unsigned length_in){
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length_in){
        vector_inout[index] -= vector_in[index];
    }
}
__device__ void _Mul(double* vector_inout, double* vector_in, unsigned length_in){
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length_in){
        vector_inout[index] *= vector_in[index];
    }
}
__device__ void _Mul(double* vector_inout, double value_in, unsigned length_in){
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length_in){
        vector_inout[index] *= value_in;
    }
}

//In-Placed Calculation
void Add(double* vector_inout, double* vector_in, unsigned length_in, cudaStream_t stream) {
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in+T.x-1u)/T.x);
    _Add<<<T,B,0,stream>>>(vector_inout, vector_in, length_in);
}
void Sub(double* vector_inout, double* vector_in, unsigned length_in, cudaStream_t stream) {
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in+T.x-1u)/T.x);
    _Sub<<<T,B,0,stream>>>(vector_inout, vector_in, length_in);
}
void Mul(double* vector_inout, double* vector_in, unsigned length_in, cudaStream_t stream) {
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in+T.x-1u)/T.x);
    _Mul<<<T,B,0,stream>>>(vector_inout, vector_in, length_in);
}
void Mul(double* vector_inout, double value_in, unsigned length_in, cudaStream_t stream) {
    unsigned length = MathGPU::properties->maxThreadsPerBlock;
    dim3 T(length);
    dim3 B((length_in+T.x-1u)/T.x);
    _Mul<<<T,B,0,stream>>>(vector_inout, value_in, length_in);
}
