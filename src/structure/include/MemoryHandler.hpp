#ifndef MEMORYHANDLER_HEADER
#define MEMORYHANDLER_HEADER

#include <list>
#include <new>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../../cuda/include/HelperCUDA.hpp"

enum PointerType
{
    CPU,
    GPU,
};

enum PointerContext
{
    CPU_Only,
    GPU_Aware,
};

enum PointerDataTransfer
{
    HostToHost,
    HostToDevice,
    DeviceToHost,
    HostAwareToDevice,
    DeviceToHostAware,
    DeviceToDevice,
};

template <typename T>
struct Pointer
{
public:
    T *pointer;
    PointerType type;
    PointerContext context;

    Pointer()
    {
        pointer = NULL;
        type = PointerType::CPU;
        context = PointerContext::CPU_Only;
    }

    Pointer(PointerType type_in, PointerContext context_in)
    {
        pointer = NULL;
        type = type_in;
        context = context_in;
    }

    Pointer(T *pointer_in, PointerType type_in, PointerContext context_in)
    {
        pointer = pointer_in;
        type = type_in;
        context = context_in;
    }
};

class MemoryHandler
{
private:
    static cudaStream_t *cudaStreams;
    static cublasHandle_t *cublasHandles;
    static cusolverDnHandle_t *cusolverHandles;
    static unsigned cudaStreams_size;
    static unsigned cublasHandles_size;
    static unsigned cusolverHandles_size;

    static bool GPUisEnabled;

    static void AllocStreams(unsigned size_in)
    {
        cudaStreams = new (std::nothrow) cudaStream_t[size_in];
        cudaStreams_size = size_in;
        for (unsigned i = 0; i < cudaStreams_size; i++)
        {
            cudaStreamCreate(&(cudaStreams[i]));
        }
    }

    static void AllocCuBLAS(unsigned size_in)
    {
        cublasHandles = new (std::nothrow) cublasHandle_t[size_in];
        cublasHandles_size = size_in;
        for (unsigned i = 0; i < cublasHandles_size; i++)
        {
            cublasCreate_v2(&(cublasHandles[i]));
        }
    }

    static void AllocCuSolverHandles(unsigned size_in)
    {
        cusolverHandles = new (std::nothrow) cusolverDnHandle_t[size_in];
        cusolverHandles_size = size_in;
        for (unsigned i = 0; i < cusolverHandles_size; i++)
        {
            cusolverDnCreate(&(cusolverHandles[i]));
        }
    }

    static void FreeStreams()
    {
        for (unsigned i = 0; i < cudaStreams_size; i++)
        {
            cudaStreamDestroy(cudaStreams[i]);
        }
        cudaStreams_size = 0u;
        delete[] cudaStreams;
    }

    static void FreeCuBLAS()
    {
        for (unsigned i = 0; i < cublasHandles_size; i++)
        {
            cublasDestroy_v2(cublasHandles[i]);
        }
        cublasHandles_size = 0u;
        delete[] cublasHandles;
    }

    static void FreeCuSolverHandles()
    {
        for (unsigned i = 0; i < cusolverHandles_size; i++)
        {
            cusolverDnDestroy(cusolverHandles[i]);
        }
        cusolverHandles_size = 0u;
        delete[] cusolverHandles;
    }

public:
    MemoryHandler()
    {
    }

    static void CreateGPUContext(unsigned cudaSize_in, unsigned cublasSize_in, unsigned cusolverSize_in)
    {
        AllocStreams(cudaSize_in);
        AllocCuBLAS(cublasSize_in);
        AllocCuSolverHandles(cusolverSize_in);
        GPUisEnabled = true;
    }

    static void DestroyGPUContext()
    {
        FreeCuSolverHandles();
        FreeCuBLAS();
        FreeStreams();
        GPUisEnabled = false;
    }

    static cudaStream_t GetStream(unsigned index_in)
    {
        return MemoryHandler::cudaStreams[index_in];
    }

    static cublasHandle_t GetCuBLASHandle(unsigned index_in)
    {
        return MemoryHandler::cublasHandles[index_in];
    }

    static cusolverDnHandle_t GetCuSolverHandle(unsigned index_in)
    {
        return MemoryHandler::cusolverHandles[index_in];
    }

    template <typename T, typename S>
    static Pointer<T> AllocValue(const S &value_in, PointerType type_in, PointerContext context_in)
    {
        Pointer<T> output;
        output.type = type_in;
        output.context = context_in;
        T *pointer_aux;
        switch (type_in)
        {
        case PointerType::CPU:
            switch (context_in)
            {
            case PointerContext::CPU_Only:
                output.pointer = new T[1u];
                output.pointer[0u] = T(value_in);
                break;
            case PointerContext::GPU_Aware:
                cudaMallocHost(&(output.pointer), sizeof(T));
                output.pointer[0u] = T(value_in);
            default:
                std::cout << "Error: Behavior of this context is not defined for this type.\n";
                break;
            }
            break;
        case PointerType::GPU:
            pointer_aux = new T(value_in);
            cudaMalloc(&(output.pointer), sizeof(T));
            cudaMemcpy(output.pointer, &pointer_aux, sizeof(T), cudaMemcpyHostToDevice);
            delete pointer_aux;
            break;
        default:
            std::cout << "Error: Behavior of this type is not defined.\n";
            break;
        }
        return output;
    }

    template <typename T>
    static Pointer<T> Alloc(unsigned size_in, PointerType type_in, PointerContext context_in, cudaStream_t stream_in = cudaStreamDefault)
    {
        if (size_in == 0u)
        {
            std::cout << "Error: size equals to 0.\n";
            return Pointer<T>();
        }
        Pointer<T> output;
        output.type = type_in;
        output.context = context_in;
        switch (type_in)
        {
        case PointerType::CPU:
            switch (context_in)
            {
            case PointerContext::CPU_Only:
                output.pointer = new (std::nothrow) T[size_in];
                break;
            case PointerContext::GPU_Aware:
                cudaMallocHost(&(output.pointer), sizeof(T) * size_in);
                break;
            default:
                std::cout << "Error: Behavior of this context is not defined for this type.\n";
                break;
            }
            break;
        case PointerType::GPU:
            cudaMalloc(&(output.pointer), sizeof(T) * size_in);
            break;
        default:
            std::cout << "Error: Behavior of this type is not defined.\n";
            break;
        }
        return output;
    }

    template <typename T>
    static void Free(Pointer<T> pointer_in, cudaStream_t stream_in = cudaStreamDefault)
    {
        if (pointer_in.pointer == NULL)
        {
            std::cout << "Error: freeing null pointer.\n";
            return;
        }
        switch (pointer_in.type)
        {
        case PointerType::CPU:
            switch (pointer_in.context)
            {
            case PointerContext::CPU_Only:
                delete[] pointer_in.pointer;
                break;
            case PointerContext::GPU_Aware:
                cudaFreeHost(pointer_in.pointer);
                break;
            default:
                std::cout << "Error: Behavior of this context is not defined for this type.\n";
                break;
            }
            break;
        case PointerType::GPU:
            cudaFreeAsync(pointer_in.pointer,stream_in);
            break;
        default:
            std::cout << "Error: Behavior of this type is not defined.\n";
            break;
        }
        pointer_in.pointer = NULL;
    }

    static PointerDataTransfer TransferType(PointerType outputType_in, PointerContext outputContext_in, PointerType inputType_in, PointerContext inputContext_in)
    {
        PointerDataTransfer transfer = PointerDataTransfer::HostToHost;
        switch (inputType_in)
        {
        case PointerType::CPU:
            if (outputType_in == inputType_in)
            {
                transfer = PointerDataTransfer::HostToHost;
            }
            else
            {
                if (inputContext_in == PointerContext::CPU_Only)
                {
                    transfer = PointerDataTransfer::HostToDevice;
                }
                else if (inputContext_in == PointerContext::GPU_Aware)
                {
                    transfer = PointerDataTransfer::HostAwareToDevice;
                }
                else
                {
                    std::cout << "Error: Behavior of this type is not defined.\n";
                }
            }
            break;
        case PointerType::GPU:
            if (outputType_in == inputType_in)
            {
                transfer = PointerDataTransfer::DeviceToDevice;
            }
            else
            {
                if (outputContext_in == PointerContext::CPU_Only)
                {
                    transfer = PointerDataTransfer::DeviceToHost;
                }
                else if (outputContext_in == PointerContext::GPU_Aware)
                {
                    transfer = PointerDataTransfer::DeviceToHostAware;
                }
                else
                {
                    std::cout << "Error: Behavior of this type is not defined.\n";
                }
            }
            break;
        default:
            std::cout << "Error: Behavior of this type is not defined.\n";
            break;
        }
        return transfer;
    }

    template <typename T>
    static void Copy(Pointer<T> pointerTo_out, Pointer<T> pointerFrom_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault)
    {
        if (pointerTo_out.pointer == NULL || pointerFrom_in.pointer == NULL)
        {
            std::cout << "Error: copying null pointers.\n";
            return;
        }
        PointerDataTransfer transfer = TransferType(pointerTo_out.type, pointerTo_out.context, pointerFrom_in.type, pointerFrom_in.context);
        switch (transfer)
        {
        case PointerDataTransfer::HostToHost:
            for (unsigned i = 0u; i < length_in; i++)
            {
                pointerTo_out.pointer[i] = pointerFrom_in.pointer[i];
            }
            break;
        case PointerDataTransfer::HostToDevice:
            cudaMemcpy(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(T) * length_in, cudaMemcpyHostToDevice);
            break;
        case PointerDataTransfer::DeviceToHost:
            cudaMemcpy(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(T) * length_in, cudaMemcpyDeviceToHost);
            break;
        case PointerDataTransfer::HostAwareToDevice:
            cudaMemcpyAsync(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(T) * length_in, cudaMemcpyHostToDevice, stream_in);
            break;
        case PointerDataTransfer::DeviceToHostAware:
            cudaMemcpyAsync(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(T) * length_in, cudaMemcpyDeviceToHost, stream_in);
            break;
        case PointerDataTransfer::DeviceToDevice:
            cudaMemcpyAsync(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(T) * length_in, cudaMemcpyDeviceToDevice, stream_in);
            break;
        default:
            break;
        }
    }

    template <typename T>
    static void Set(Pointer<T> pointer_inout, const T &value, unsigned start, unsigned end)
    {
        if (pointer_inout.pointer == NULL)
        {
            std::cout << "Error: set null pointer.\n";
            return;
        }
        switch (pointer_inout.type)
        {
        case PointerType::CPU:
            switch (pointer_inout.context)
            {
            case PointerContext::CPU_Only:
                for (unsigned i = start; i < end; i++)
                {
                    pointer_inout.pointer[i] = value;
                }
                break;
            case PointerContext::GPU_Aware:
                for (unsigned i = start; i < end; i++)
                {
                    pointer_inout.pointer[i] = value;
                }
            default:
                std::cout << "Error: Behavior of this context is not defined for this type.\n";
                break;
            }
            break;
        case PointerType::GPU:
            HelperCUDA::Initialize(pointer_inout.pointer + start, &value, end - start, sizeof(T));
            break;
        default:
            std::cout << "Error: Behavior of this type is not defined.\n";
            break;
        }
    }
};

template <>
Pointer<void> MemoryHandler::Alloc(unsigned size_in, PointerType type_in, PointerContext context_in, cudaStream_t stream_in);
template <>
void MemoryHandler::Free(Pointer<void> pointer_in, cudaStream_t stream_in);
template <>
void MemoryHandler::Copy(Pointer<void> pointerTo_out, Pointer<void> pointerFrom_in, unsigned length_in, cudaStream_t stream_in);
    
#endif