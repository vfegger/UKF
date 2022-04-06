#include "../include/Parameter.hpp"

Parameter::Parameter(unsigned lengthElements_in){
    lengthElements = lengthElements_in;
    names = new(std::nothrow) std::string[lengthElements];
    sizeTypeArray = new(std::nothrow) unsigned[lengthElements];
    lengthArray = new(std::nothrow) unsigned[lengthElements];
    offsetPointer = new(std::nothrow) void*[lengthElements];
    for(unsigned i = 0; i < lengthElements; i++){
        names[i] = "";
        sizeTypeArray[i] = 0u;
        lengthArray[i] = 0u;
        offsetPointer[i] = NULL;
    }
    pointer = NULL;
    length = 0u;
    isValid = false;
    count = 0u;
}
Parameter::Parameter(const Parameter& parameter_in){
    lengthElements = parameter_in.lengthElements;
    names = new(std::nothrow) std::string[lengthElements];
    sizeTypeArray = new(std::nothrow) unsigned[lengthElements];
    lengthArray = new(std::nothrow) unsigned[lengthElements];
    offsetPointer = new(std::nothrow) void*[lengthElements];
    for(unsigned i = 0; i < lengthElements; i++){
        names[i] = parameter_in.names[i];
        sizeTypeArray[i] = parameter_in.sizeTypeArray[i];
        lengthArray[i] = parameter_in.lengthArray[i];
        offsetPointer[i] = NULL;
    }
    pointer = NULL;
    length = 0u;
    count = parameter_in.count;
    isValid = parameter_in.isValid;
    if(isValid){
        length = parameter_in.length;
        pointer = (void*)new(std::nothrow) char[length];
        for(unsigned i = 0u; i < length; i++){
            ((char*)pointer)[i] = ((char*)parameter_in.pointer)[i];
        }
        unsigned offset_aux = 0u;
        for(unsigned i = 0; i < lengthElements; i++){
            offsetPointer[i] = (void*)((char*)pointer + offset_aux);
            offset_aux += lengthArray[i] * sizeTypeArray[i];
        }
    }
}
unsigned Parameter::Add(std::string name_in, unsigned length_in, unsigned sizeType_in){
    isValid = false;
    if(count >= lengthElements){
        std::cout << "Error: Added element is over the limit.";
        return lengthElements;
    }
    names[count] = name_in;
    sizeTypeArray[count] = sizeType_in;
    lengthArray[count] = length_in;
    count++;
    return count-1;
}
void Parameter::Add(unsigned* indexes, std::string* names_in, unsigned* lengthArray_in, unsigned* sizeTypeArray_in, unsigned lengthElements_in){
    unsigned index = 0u;
    for(unsigned i = 0; i < lengthElements_in; i++){
        index = Add(names_in[i], lengthArray_in[i],sizeTypeArray_in[i]);
        if(indexes != NULL){
            indexes[i] = index;
        }
    }
}
void Parameter::Initialize(){
    if(pointer != NULL){
        delete[] (char*)pointer;
    }
    for(unsigned i = 0u; i < count; i++){
        length += lengthArray[i] * sizeTypeArray[i];
    }
    pointer = (void*)new(std::nothrow) char[length];
    if(pointer == NULL){
        std::cout << "Error: Initialization wasn't successful.";
        return;
    }
    unsigned offset_aux = 0u;
    for(unsigned i = 0u; i < count; i++){
        offsetPointer[i] = (void*)((char*)pointer + offset_aux);
        offset_aux += lengthArray[i] * sizeTypeArray[i];
    }
    isValid = true;
}
unsigned Parameter::GetLength(unsigned index){
    if(isValid == false){
        std::cout << "Error: Length of pointer is not initialized.";
        return 0u;
    }
    return lengthArray[index];
}
bool Parameter::GetValidation(){
    return isValid;
}
Parameter::~Parameter(){
    if(pointer != NULL){
        delete[] (char*)pointer;
    }
    delete[] offsetPointer;
    delete[] lengthArray;
    delete[] sizeTypeArray;
    delete[] names;
}