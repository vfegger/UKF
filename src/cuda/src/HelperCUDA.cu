#include "../include/HelperCUDA.hpp"

// TODO: Initialize Vector in CUDA

#include <iostream>

__global__ void InitializeVector(void *pointer_out, const void *value_in, unsigned length_in, unsigned size_in)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    char *p_in = (char *)value_in;
    char *p_out = (char *)pointer_out + index * size_in;
    if (index < length_in)
    {
        for (unsigned i = 0u; i < size_in; i++)
        {
            p_out[i] = p_in[i];
        }
    }
}

void HelperCUDA::Initialize(void *pointer_out, const void *value_in, unsigned length_in, unsigned size_in)
{
    dim3 T(1024u);
    dim3 B((length_in + T.x - 1u) / T.x);
    void *value_aux = NULL;
    cudaMallocAsync(&value_aux, size_in, cudaStreamDefault);
    cudaMemcpyAsync(value_aux, value_in, size_in, cudaMemcpyHostToDevice, cudaStreamDefault);
    InitializeVector<<<B, T, 0, cudaStreamDefault>>>(pointer_out, value_aux, length_in, size_in);
    cudaFreeAsync(value_aux, cudaStreamDefault);
    cudaStreamSynchronize(cudaStreamDefault);
}