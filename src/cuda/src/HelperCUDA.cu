#include "../include/HelperCUDA.hpp"

__global__ void InitializeVector(void *pointer_out, void *value_in, unsigned length_in, unsigned size_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned size = size_in / sizeof(char);
    if (index < length_in)
    {
        for (unsigned i = 0u; i < size; i++)
        {
            ((char *)pointer_out)[i] = ((char *)value_in)[i];
        }
    }
}

void HelperCUDA::Initialize(void *pointer_out, void *value_in, unsigned length_in, unsigned size_in)
{
    dim3 T(1024u);
    dim3 B((length_in + T.x - 1u) / T.x);
    void * value_aux;
    cudaMalloc(&value_aux,size_in);
    cudaMemcpy(value_aux,value_in,size_in,cudaMemcpyHostToDevice);
    InitializeVector<<<B, T, 0, 0>>>(pointer_out, value_in, length_in, size_in);
    cudaFree(value_aux);
}
