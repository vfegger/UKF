#include <list>
#include <new>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


enum PointerType {
    CPU, GPU,
};

enum PointerContext {
    CPU_Only, GPU_Aware,
};

template<typename T>
struct Pointer {
public:
    T* pointer;
    PointerType type;
    PointerContext context;
};

class MemoryHandler {
private:
    cudaStream_t* cudaStreams;
    cublasHandle_t* cublasHandles;
    cusolverDnHandle_t* cusolverHandles;
public:
    MemoryHandler(){
        cudaStreams = NULL;
        cublasHandles = NULL;
        cusolverHandles = NULL;
    }

    template<typename T>
    static Pointer<T> Alloc(unsigned size_in, PointerType type_in, PointerContext context_in){
        Pointer<T> output;
        output.type = type_in;
        output.context = context_in;
        switch (type_in)
        {
        case PointerType::CPU:
            switch (context_in)
            {
            case PointerContext::CPU_Only:
                output.pointer = new(std::nothrow) T[size];
                break;
            case PointerContext::GPU_Aware:
                cudaMallocHost(&(output.pointer),sizeof(T)*size_in);
            default:
                std::cout << "Error: Behavior of this context is not defined for this type.\n";
                break;
            }
            break;
        case PointerType::GPU:
            cudaMalloc(&(output.pointer),sizeof(T)*size_in);
            break;
        default:
            std::cout << "Error: Behavior of this type is not defined.\n";
            break;
        }
        return output;
    }

    template<typename T>
    static void Free(Pointer<T> pointer_in){
        switch (pointer_in.type)
        {
        case PointerType::CPU:
            switch (pointer_in.context)
            {
            case PointerContext::CPU_Only:
                delete[] pointer_in.pointer;
                break;
            case PointerContext::GPU_Aware:
                cudaFreeHost(output.pointer);
            default:
                std::cout << "Error: Behavior of this context is not defined for this type.\n";
                break;
            }
            break;
        case PointerType::GPU:
            cudaFree(output.pointer);
            break;
        default:
            std::cout << "Error: Behavior of this type is not defined.\n";
            break;
        }
    }
};
