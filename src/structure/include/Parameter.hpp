#ifndef PARAMETER_HEADER
#define PARAMETER_HEADER

#include <string>
#include <iostream>
#include <new>

class Parameter{
private:
    void* pointer;
    unsigned length;
    
    void** offsetPointer;
    unsigned* sizeTypeArray;
    unsigned* lengthArray;
    unsigned lengthElements;

    std::string* names;

    bool isValid;
    unsigned count;
public:
    Parameter(unsigned lengthElements_in);
    Parameter(const Parameter& parameter_in);
    unsigned Add(std::string name_in, unsigned length_in, unsigned sizeType_in);
    void Add(unsigned* indexes, std::string* names_in, unsigned* lengthArray_in, unsigned* sizeTypeArray_in, unsigned lengthElements_in);
    void Initialize();
    template<typename T>
    void LoadData(unsigned index_in, T* array_in, unsigned length_in);
    template<typename T>
    void LoadData(unsigned* indexes_in, T** array_in, unsigned* lengthArray_in, unsigned lengthElements_in);
    template<typename T>
    T* GetPointer(unsigned index);
    unsigned GetLength(unsigned index);
    bool GetValidation();
    ~Parameter();
};


template<typename T>
void Parameter::LoadData(unsigned index_in, T* array_in, unsigned length_in){
    if(isValid == false){
        std::cout << "Error: Load while structure is not initialized.";
        return;
    }
    if(index_in >= count){
        std::cout << "Error: Index is out of range.";
        return;
    }
    for(unsigned i = 0; i < length_in; i++){
        ((T*)(offsetPointer[index_in]))[i] = array_in[i];
    }
}

template<typename T>
void Parameter::LoadData(unsigned* indexes_in, T** array_in, unsigned* lengthArray_in, unsigned lengthElements_in){
    for(unsigned i = 0u; i < lengthElements_in; i++){
        LoadData(indexes_in[i], array_in[i], lengthArray_in[i]);
    }
}

template<typename T>
T* Parameter::GetPointer(unsigned index){
    if(isValid == false){
        std::cout << "Error: Pointer is not initialized.";
        return NULL;
    }
    return (T*)(offsetPointer[index]);
}

#endif