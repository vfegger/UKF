#include "../include/Parameter.hpp"

Parameter::Parameter()
{
    lengthElements = 0;
    names = Pointer<std::string>(PointerType::CPU, PointerContext::CPU_Only);
    lengthArray = Pointer<unsigned>(PointerType::CPU, PointerContext::CPU_Only);
    sizeTypeArray = Pointer<unsigned>(PointerType::CPU,PointerContext::CPU_Only);
    offset = Pointer<Pointer<void>>(PointerType::CPU, PointerContext::CPU_Only);
    pointer = Pointer<void>();
    length = 0u;
    isValid = false;
    count = 0u;
}
Parameter::Parameter(unsigned lengthElements_in){
    lengthElements = lengthElements_in;
    names = MemoryHandler::Alloc<std::string>(lengthElements,PointerType::CPU,PointerContext::CPU_Only);
    sizeTypeArray = MemoryHandler::Alloc<unsigned>(lengthElements,PointerType::CPU,PointerContext::CPU_Only);
    lengthArray = MemoryHandler::Alloc<unsigned>(lengthElements,PointerType::CPU,PointerContext::CPU_Only);
    offset = MemoryHandler::Alloc<Pointer<void>>(lengthElements,PointerType::CPU,PointerContext::CPU_Only);
    for(unsigned i = 0; i < lengthElements; i++){
        names.pointer[i] = "";
        sizeTypeArray.pointer[i] = 0u;
        lengthArray.pointer[i] = 0u;
        offset.pointer[i] = Pointer<void>();
    }
    pointer =  Pointer<void>();
    length = 0u;
    isValid = false;
    count = 0u;
}
Parameter::Parameter(const Parameter& parameter_in){
    lengthElements = parameter_in.lengthElements;
    names = MemoryHandler::Alloc<std::string>(lengthElements,PointerType::CPU,PointerContext::CPU_Only);
    sizeTypeArray = MemoryHandler::Alloc<unsigned>(lengthElements,PointerType::CPU,PointerContext::CPU_Only);
    lengthArray = MemoryHandler::Alloc<unsigned>(lengthElements,PointerType::CPU,PointerContext::CPU_Only);
    offset = MemoryHandler::Alloc<Pointer<void>>(lengthElements,PointerType::CPU,PointerContext::CPU_Only);
    for(unsigned i = 0; i < lengthElements; i++){
        names.pointer[i] = parameter_in.names.pointer[i];
        sizeTypeArray.pointer[i] = parameter_in.sizeTypeArray.pointer[i];
        lengthArray.pointer[i] = parameter_in.lengthArray.pointer[i];
        offset.pointer[i] = Pointer<void>(parameter_in.offset.type, parameter_in.offset.context);
    }
    pointer = Pointer<void>();
    length = 0u;
    count = parameter_in.count;
    isValid = parameter_in.isValid;
    if(isValid){
        length = parameter_in.length;
        pointer = MemoryHandler::Alloc<void>(length,parameter_in.pointer.type,parameter_in.pointer.context);
        for(unsigned i = 0u; i < length; i++){
            ((char*)pointer.pointer)[i] = ((char*)parameter_in.pointer.pointer)[i];
        }
        unsigned offset_aux = 0u;
        for(unsigned i = 0; i < lengthElements; i++){
            offset.pointer[i] = Pointer<void>(((char*)pointer.pointer + offset_aux),parameter_in.pointer.type,parameter_in.pointer.context);
            offset_aux += lengthArray.pointer[i] * sizeTypeArray.pointer[i];
        }
    }
}
unsigned Parameter::Add(std::string name_in, unsigned length_in, unsigned sizeType_in){
    isValid = false;
    if(count >= lengthElements){
        std::cout << "Error: Added element is over the limit.";
        return lengthElements;
    }
    names.pointer[count] = name_in;
    sizeTypeArray.pointer[count] = sizeType_in;
    lengthArray.pointer[count] = length_in;
    count++;
    return count-1;
}
void Parameter::Add(Pointer<unsigned> indexes, Pointer<std::string> names_in, Pointer<unsigned> lengthArray_in, Pointer<unsigned> sizeTypeArray_in, unsigned lengthElements_in){
    unsigned index = 0u;
    for(unsigned i = 0; i < lengthElements_in; i++){
        index = Add(names_in.pointer[i], lengthArray_in.pointer[i],sizeTypeArray_in.pointer[i]);
        if(indexes.pointer != NULL){
            indexes.pointer[i] = index;
        }
    }
}
void Parameter::Initialize(PointerType type_in, PointerContext context_in){
    if(pointer.pointer != NULL){
        MemoryHandler::Free(pointer);
    }
    for(unsigned i = 0u; i < count; i++){
        length += lengthArray.pointer[i] * sizeTypeArray.pointer[i];
    }
    pointer = MemoryHandler::Alloc<void>(length,type_in,context_in);
    if(pointer.pointer == NULL){
        std::cout << "Error: Initialization wasn't successful.";
        return;
    }
    unsigned offset_aux = 0u;
    for(unsigned i = 0u; i < count; i++){
        offset.pointer[i] = Pointer<void>((void*)((char*)pointer.pointer + offset_aux),type_in,context_in);
        offset_aux += lengthArray.pointer[i] * sizeTypeArray.pointer[i];
    }
    isValid = true;
}
unsigned Parameter::GetLength(unsigned index){
    if(isValid == false){
        std::cout << "Error: Length of pointer is not initialized.";
        return 0u;
    }
    return lengthArray.pointer[index];
}
bool Parameter::GetValidation(){
    return isValid;
}
Parameter::~Parameter(){
    if(pointer.pointer != NULL){
        MemoryHandler::Free(pointer);
    }
    MemoryHandler::Free(offset);
    MemoryHandler::Free(lengthArray);
    MemoryHandler::Free(sizeTypeArray);
    MemoryHandler::Free(names);
}