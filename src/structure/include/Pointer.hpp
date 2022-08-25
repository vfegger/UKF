#ifndef POINTER_HEADER
#define POINTER_HEADER

#include <iostream>
#include <new>

#include "cuda.h"

enum PointerType {
    None, CPU, GPU
};

template<typename T>
class Pointer {
private:
    T* pointer;
    PointerType type;
    long long unsigned size;
    bool isAllocated;
public:
    //Base Constructor
    Pointer(){
        pointer = NULL;
        type = PointerType::None;
        isAllocated = false;
    }

    //Wrapper Constructor
    Pointer(T* pointer_in, long long unsigned size_in, PointerType type_in = PointerType::CPU){
        pointer = pointer_in;
        type = type_in;
        size = size_in;
        isAllocated = false;
    }

    //TODO: Pointer to Pointer Constructor/Copy/Assign
    Pointer(Pointer& pointer_in){
        pointer = pointer_in.pointer;
        type = pointer_in.type;
        size = pointer_in.size;
        isAllocated = false;
    }

    //Pointer Functions
    
    //Get Pointer
    T* GetPointer() {
        return pointer;
    }

    void SetPointer(T* pointer_in, long long unsigned size_in, PointerType type_in = PointerType::CPU){
        pointer = pointer_in;
        type = type_in;
        size = size_in;
        isAllocated = false;
    }

    //Alloc pointer memory of a given type and size using the class
    T* Alloc(long long unsigned size_in, PointerType type_in = PointerType::CPU){
        if(isAllocated){
            std::cout << "Warning: Trying to alloc a second time, freeing the first allocated memory."
            Free();
        }
        size = size_in;
        isAllocated = true;
        type = type_in;
        switch (type)
        {
        case PointerType::CPU:
            pointer = new T[1llu];
            break;
        case PointerType::GPU:
            cudaMalloc(&pointer,size*sizeof(T));
            break;
        default:
            break;
        }
    }

    //Free pointer memory of a given type and size using the class
    void Free(){
        if(isAllocated){
            switch (type)
            {
            case PointerType::CPU:
                delete[] pointer;
                pointer = NULL;
                break;
            case PointerType::GPU:
                cudaFree(pointer);
                pointer = NULL;
                break;
            default:
                break;
            }
        }
        size = 0llu;
        isAllocated = false;
        type = PointerType::None;
    }

};

#endif