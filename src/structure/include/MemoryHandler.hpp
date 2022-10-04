#include <list>
#include <new>
#include <iostream>
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
    unsigned cudaStreams_size;
    cublasHandle_t* cublasHandles;
    unsigned cublasHandles_size;
    unsigned cublasHandles_count;
    cusolverDnHandle_t* cusolverHandles;
    unsigned cusolverHandles_size;
    unsigned cusolverHandles_count;
public:
    MemoryHandler(){
        cudaStreams = NULL;
        cudaStreams_size = 0u;
        cublasHandles = NULL;
        cublasHandles_size = 0u;
        cublasHandles_count = 0u;
        cusolverHandles = NULL;
        cusolverHandles_size = 0u;
        cusolverHandles_count = 0u;
    }

    void AllocStreams(unsigned size_in){
        cudaStreams = new(std::nothrow) cudaStream_t[size_in];
        cudaStreams_size = size_in;
        for(unsigned i = 0; i < cudaStreams_size; i++){
            cudaStreamCreate(&(cudaStreams[i]));
        }
    }

    void FreeStreams(){
        for(unsigned i = 0; i < cudaStreams_size; i++){
            cudaStreamSynchronize(cudaStreams[i]);
            cudaStreamDestroy(cudaStreams[i]);
        }
        cudaStreams_size = 0u;
        delete[] cudaStreams;
    }

    void AllocCuBLASHandles(unsigned size_in){
        cublasHandles = new(std::nothrow) cublasHandle_t[size_in];
        cublasHandles_size = size_in;
        cublasHandles_count = 0u;
    }

    void CreateCuBLASHandle(unsigned index_in){
        if(index_in >= cublasHandles_size) {
            std::cout << "Error: CuBLAS Handle cannot be created.\n";
            return;
        }
        cublasCreate_v2(&(cublasHandles[index_in]));
        return;
    }

    void DestroyCuBLASHandle(unsigned index_in){
        if(index_in >= cublasHandles_size) {
            std::cout << "Error: CuBLAS Handle cannot be destroyed.\n";
            return;
        }
        cublasDestroy_v2(cublasHandles[index_in]);
        return;
    }

    void FreeCuBLASHandles(){
        cublasHandles_size = 0u;
        delete[] cublasHandles;
    }

    void AllocCuSolverHandles(unsigned size_in){
        cusolverHandles = new(std::nothrow) cusolverDnHandle_t[size_in];
        cusolverHandles_size = size_in;
        cusolverHandles_count = 0u;
    }

    void CreateCuSolverHandle(unsigned index_in){
        if(index_in >= cusolverHandles_size) {
            std::cout << "Error: CuSolver Handle cannot be created.\n";
            return;
        }
        cusolverDnCreate(&(cusolverHandles[index_in]));
        return;
    }

    void DestroyCuSolverHandle(unsigned index_in){
        if(index_in >= cublasHandles_size) {
            std::cout << "Error: CuSolver Handle cannot be destroyed.\n";
            return;
        }
        cusolverDnDestroy(cusolverHandles[index_in]);
        return;
    }

    void FreeCuSolverHandles(){
        cublasHandles_size = 0u;
        delete[] cublasHandles;
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