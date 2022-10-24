#include "../include/MemoryHandler.hpp"

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
        cudaDeviceSynchronize();
        cudaFree((char*)(pointer_in.pointer));
        break;
    default:
        std::cout << "Error: Behavior of this type is not defined.\n";
        break;
    }
    pointer_in.pointer = NULL;
}