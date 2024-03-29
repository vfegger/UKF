#include "../include/MemoryHandler.hpp"

// Static variables
cudaStream_t *MemoryHandler::cudaStreams = NULL;
cublasHandle_t *MemoryHandler::cublasHandles = NULL;
cusolverDnHandle_t *MemoryHandler::cusolverHandles = NULL;
unsigned MemoryHandler::cudaStreams_size = 0u;
unsigned MemoryHandler::cublasHandles_size = 0u;
unsigned MemoryHandler::cusolverHandles_size = 0u;
bool MemoryHandler::GPUisEnabled = false;


cublasHandle_t MemoryHandler::cublasHandle_aux = NULL;
cusolverDnHandle_t MemoryHandler::cusolverHandle_aux = NULL;

// Void explicit specialization

template <>
Pointer<void> MemoryHandler::Alloc<void>(unsigned size_in, PointerType type_in, PointerContext context_in, cudaStream_t stream_in)
{
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
            output.pointer = (void *)(new (std::nothrow) char[size_in]);
            break;
        case PointerContext::GPU_Aware:
            cudaMallocHost(&(output.pointer), sizeof(char) * size_in);
        default:
            std::cout << "Error: Behavior of this context is not defined for this type.\n";
            break;
        }
        break;
    case PointerType::GPU:
        cudaMallocAsync(&(output.pointer), sizeof(char) * size_in, stream_in);
        break;
    default:
        std::cout << "Error: Behavior of this type is not defined.\n";
        break;
    }
    return output;
}

template <>
void MemoryHandler::Free(Pointer<void> pointer_in, cudaStream_t stream_in)
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
            delete[](char *)(pointer_in.pointer);
            break;
        case PointerContext::GPU_Aware:
            cudaFreeHost((char *)(pointer_in.pointer));
        default:
            std::cout << "Error: Behavior of this context is not defined for this type.\n";
            break;
        }
        break;
    case PointerType::GPU:
        cudaStreamSynchronize(stream_in);
        cudaFreeAsync((char *)(pointer_in.pointer), stream_in);
        break;
    default:
        std::cout << "Error: Behavior of this type is not defined.\n";
        break;
    }
    pointer_in.pointer = NULL;
}

template <>
void MemoryHandler::Copy(Pointer<void> pointerTo_out, Pointer<void> pointerFrom_in, unsigned length_in, cudaStream_t stream_in)
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
            ((char *)pointerTo_out.pointer)[i] = ((char *)pointerFrom_in.pointer)[i];
        }
        break;
    case PointerDataTransfer::HostToDevice:
        cudaMemcpy(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(char) * length_in, cudaMemcpyHostToDevice);
        break;
    case PointerDataTransfer::DeviceToHost:
        cudaMemcpy(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(char) * length_in, cudaMemcpyDeviceToHost);
        break;
    case PointerDataTransfer::HostAwareToDevice:
        cudaMemcpyAsync(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(char) * length_in, cudaMemcpyHostToDevice, stream_in);
        break;
    case PointerDataTransfer::DeviceToHostAware:
        cudaMemcpyAsync(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(char) * length_in, cudaMemcpyDeviceToHost, stream_in);
        break;
    case PointerDataTransfer::DeviceToDevice:
        cudaMemcpyAsync(pointerTo_out.pointer, pointerFrom_in.pointer, sizeof(char) * length_in, cudaMemcpyDeviceToDevice, stream_in);
        break;
    default:
        break;
    }
}
