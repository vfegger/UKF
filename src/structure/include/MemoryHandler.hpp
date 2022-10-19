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
    cudaStream_t *cudaStreams;
    unsigned cudaStreams_size;
    cublasHandle_t *cublasHandles;
    unsigned cublasHandles_size;
    unsigned cublasHandles_count;
    cusolverDnHandle_t *cusolverHandles;
    unsigned cusolverHandles_size;
    unsigned cusolverHandles_count;

public:
    MemoryHandler()
    {
        cudaStreams = NULL;
        cudaStreams_size = 0u;
        cublasHandles = NULL;
        cublasHandles_size = 0u;
        cublasHandles_count = 0u;
        cusolverHandles = NULL;
        cusolverHandles_size = 0u;
        cusolverHandles_count = 0u;
    }

    void AllocStreams(unsigned size_in)
    {
        cudaStreams = new (std::nothrow) cudaStream_t[size_in];
        cudaStreams_size = size_in;
        for (unsigned i = 0; i < cudaStreams_size; i++)
        {
            cudaStreamCreate(&(cudaStreams[i]));
        }
    }

    void FreeStreams()
    {
        for (unsigned i = 0; i < cudaStreams_size; i++)
        {
            cudaStreamSynchronize(cudaStreams[i]);
            cudaStreamDestroy(cudaStreams[i]);
        }
        cudaStreams_size = 0u;
        delete[] cudaStreams;
    }

    void AllocCuBLASHandles(unsigned size_in)
    {
        cublasHandles = new (std::nothrow) cublasHandle_t[size_in];
        cublasHandles_size = size_in;
        cublasHandles_count = 0u;
    }

    void CreateCuBLASHandle(unsigned index_in)
    {
        if (index_in >= cublasHandles_size)
        {
            std::cout << "Error: CuBLAS Handle cannot be created.\n";
            return;
        }
        cublasCreate_v2(&(cublasHandles[index_in]));
        return;
    }

    void DestroyCuBLASHandle(unsigned index_in)
    {
        if (index_in >= cublasHandles_size)
        {
            std::cout << "Error: CuBLAS Handle cannot be destroyed.\n";
            return;
        }
        cublasDestroy_v2(cublasHandles[index_in]);
        return;
    }

    void FreeCuBLASHandles()
    {
        cublasHandles_size = 0u;
        delete[] cublasHandles;
    }

    void AllocCuSolverHandles(unsigned size_in)
    {
        cusolverHandles = new (std::nothrow) cusolverDnHandle_t[size_in];
        cusolverHandles_size = size_in;
        cusolverHandles_count = 0u;
    }

    void CreateCuSolverHandle(unsigned index_in)
    {
        if (index_in >= cusolverHandles_size)
        {
            std::cout << "Error: CuSolver Handle cannot be created.\n";
            return;
        }
        cusolverDnCreate(&(cusolverHandles[index_in]));
        return;
    }

    void DestroyCuSolverHandle(unsigned index_in)
    {
        if (index_in >= cublasHandles_size)
        {
            std::cout << "Error: CuSolver Handle cannot be destroyed.\n";
            return;
        }
        cusolverDnDestroy(cusolverHandles[index_in]);
        return;
    }

    void FreeCuSolverHandles()
    {
        cublasHandles_size = 0u;
        delete[] cublasHandles;
    }

    template <typename T, typename S>
    static Pointer<T> AllocValue(const S& value_in, PointerType type_in, PointerContext context_in)
    {
        Pointer<T> output;
        output.type = type_in;
        output.context = context_in;
        T* pointer_aux;
        switch (type_in)
        {
        case PointerType::CPU:
            switch (context_in)
            {
            case PointerContext::CPU_Only:
                output.pointer = new (std::nothrow) T[1];
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
            cudaMemcpy(output.pointer,&pointer_aux,sizeof(T),cudaMemcpyHostToDevice);
            delete pointer_aux;
            break;
        default:
            std::cout << "Error: Behavior of this type is not defined.\n";
            break;
        }
        return output;
    }

    template <typename T>
    static Pointer<T> Alloc(unsigned size_in, PointerType type_in, PointerContext context_in)
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
    static void Free(Pointer<T> pointer_in)
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
            default:
                std::cout << "Error: Behavior of this context is not defined for this type.\n";
                break;
            }
            break;
        case PointerType::GPU:
            cudaFree(pointer_in.pointer);
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
    static void Copy(Pointer<T> pointerTo_out, Pointer<T> pointerFrom_in, unsigned length_in)
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
            cudaMemcpyAsync(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(T) * length_in, cudaMemcpyHostToDevice);
            break;
        case PointerDataTransfer::DeviceToHostAware:
            cudaMemcpyAsync(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(T) * length_in, cudaMemcpyDeviceToHost);
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
            HelperCUDA::Initialize((pointer_inout.pointer + start), &value, end - start, sizeof(double));
            break;
        default:
            std::cout << "Error: Behavior of this type is not defined.\n";
            break;
        }
    }
};


// Void explicit specialization
template <>
Pointer<void> MemoryHandler::Alloc<void>(unsigned size_in, PointerType type_in, PointerContext context_in){
    if (size_in == 0u)
    {
        std::cout << "Error: size equals to 0.\n";
        return Pointer<void>();
    }
    Pointer<void> output;
    output.type = type_in;
    output.context = context_in;
    switch (type_in)
    {
    case PointerType::CPU:
        switch (context_in)
        {
        case PointerContext::CPU_Only:
            output.pointer = (void*)(new (std::nothrow) char[size_in]);
            break;
        case PointerContext::GPU_Aware:
            cudaMallocHost(&(output.pointer), sizeof(char) * size_in);
        default:
            std::cout << "Error: Behavior of this context is not defined for this type.\n";
            break;
        }
        break;
    case PointerType::GPU:
        cudaMalloc(&(output.pointer), sizeof(char) * size_in);
        break;
    default:
        std::cout << "Error: Behavior of this type is not defined.\n";
        break;
    }
    return output;
}

template <>
void MemoryHandler::Free(Pointer<void> pointer_in)
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
            delete[] (char*)(pointer_in.pointer);
            break;
        case PointerContext::GPU_Aware:
            cudaFreeHost((char*)(pointer_in.pointer));
        default:
            std::cout << "Error: Behavior of this context is not defined for this type.\n";
            break;
        }
        break;
    case PointerType::GPU:
        cudaFree((char*)(pointer_in.pointer));
        break;
    default:
        std::cout << "Error: Behavior of this type is not defined.\n";
        break;
    }
    pointer_in.pointer = NULL;
}

#endif