#include "../include/Data.hpp"

Data::Data(){
    lengthElements = 0;
    names = NULL;
    lengthArray = NULL;
    offsetPointer = NULL;
    pointer = NULL;
    length = 0u;
    isValid = false;
    isMultiple = false;
    indexMultiple = 0u;
    count = 0u;
}
Data::Data(unsigned lengthElements_in){
    lengthElements = lengthElements_in;
    names = new(std::nothrow) std::string[lengthElements];
    lengthArray = new(std::nothrow) unsigned[lengthElements];
    offsetPointer = new(std::nothrow) double*[lengthElements];
    for(unsigned i = 0; i < lengthElements; i++){
        names[i] = "";
        lengthArray[i] = 0u;
        offsetPointer[i] = NULL;
    }
    pointer = NULL;
    length = 0u;
    isValid = false;
    isMultiple = false;
    indexMultiple = 0u;
    count = 0u;
}
Data::Data(const Data& data_in){
    lengthElements = data_in.lengthElements;
    names = new(std::nothrow) std::string[lengthElements];
    lengthArray = new(std::nothrow) unsigned[lengthElements];
    offsetPointer = new(std::nothrow) double*[lengthElements];
    for(unsigned i = 0; i < lengthElements; i++){
        names[i] = data_in.names[i];
        lengthArray[i] = data_in.lengthArray[i];
        offsetPointer[i] = NULL;
    }
    pointer = NULL;
    length = 0u;
    count = data_in.count;
    isValid = data_in.isValid;
    isMultiple = false;
    indexMultiple = 0u;
    if(isValid){
        length = data_in.length;
        pointer = new(std::nothrow) double[length];
        for(unsigned i = 0u; i < length; i++){
            pointer[i] = data_in.pointer[i];
        }
        unsigned offset_aux = 0u;
        for(unsigned i = 0; i < lengthElements; i++){
            offsetPointer[i] = pointer + offset_aux;
            offset_aux += lengthArray[i];
        }
    }
}
unsigned Data::Add(std::string name_in, unsigned length_in){
    isValid = false;
    if(count >= lengthElements){
        std::cout << "Error: Added element is over the limit.";
        return lengthElements;
    }
    names[count] = name_in;
    lengthArray[count] = length_in;
    count++;
    return count-1;
}
void Data::Add(unsigned* indexes, std::string* names_in, unsigned* lengthArray_in, unsigned lengthElements_in){
    unsigned index = 0u;
    for(unsigned i = 0; i < lengthElements_in; i++){
        index = Add(names_in[i], lengthArray_in[i]);
        if(indexes != NULL){
            indexes[i] = index;
        }
    }
}
void Data::Initialize(){
    DeletePointer();
    length = 0u;
    for(unsigned i = 0u; i < count; i++){
        length += lengthArray[i];
    }
    pointer = new(std::nothrow) double[length];
    if(pointer == NULL){
        std::cout << "Error: Initialization wasn't successful.";
        return;
    }
    for(unsigned i = 0u; i < length; i++)
    {
        pointer[i] = 0u;
    }
    unsigned offset_aux = 0u;
    for(unsigned i = 0u; i < count; i++){
        offsetPointer[i] = pointer + offset_aux;
        offset_aux += lengthArray[i];
    }
    isValid = true;
}
void Data::InitializeMultiple(Data* dataArray_in, unsigned length_in){
    for(unsigned i = 0u; i < length_in; i++){
        dataArray_in[i].DeletePointer();
    }
    unsigned length = 0u;
    for(unsigned i = 0u; i < length_in; i++){
        dataArray_in[i].length = 0u;
        for(unsigned j = 0u; j < dataArray_in[i].count; j++){
            dataArray_in[i].length += dataArray_in[i].lengthArray[j];
        }
        length += dataArray_in[i].length;
    }
    double* pointer = new(std::nothrow) double[length];
    if(pointer == NULL){
        std::cout << "Error: Initialization of Multiple Data wasn't successful.";
        return;
    }
    unsigned offset_aux = 0u;
    for(unsigned i = 0u; i < length_in; i++){
        dataArray_in[i].pointer = pointer + offset_aux;
        for(unsigned j = 0u; j < dataArray_in[i].count; j++){
            dataArray_in[i].offsetPointer[j] = pointer + offset_aux;
            offset_aux += dataArray_in[i].lengthArray[j];
        }
        dataArray_in[i].isValid = true;
        dataArray_in[i].isMultiple = true;
        dataArray_in[i].indexMultiple = i;
    }
}

void Data::LoadData(unsigned index_in, double* array_in, unsigned length_in){
    if(isValid == false){
        std::cout << "Error: Load while structure is not initialized.";
        return;
    }
    if(index_in >= count){
        std::cout << "Error: Index is out of range.";
        return;
    }
    for(unsigned i = 0; i < length_in; i++){
        offsetPointer[index_in][i] = array_in[i];
    }
}
void Data::LoadData(unsigned* indexes_in, double** array_in, unsigned* lengthArray_in, unsigned lengthElements_in){
    for(unsigned i = 0u; i < lengthElements_in; i++){
        LoadData(indexes_in[i], array_in[i], lengthArray_in[i]);
    }
}
double* Data::GetPointer(){
    if(isValid == false){
        std::cout << "Error: Pointer is not initialized.";
        return NULL;
    }
    return pointer;
}
unsigned Data::GetLength(){
    if(isValid == false){
        std::cout << "Error: Length of pointer is not initialized.";
        return 0u;
    }
    return length;
}
bool Data::GetValidation(){
    return isValid;
}
double*& Data::operator[](unsigned index){
    return offsetPointer[index];
}
void Data::DeletePointer(){
    if(pointer != NULL){
        if(isMultiple){
            if(indexMultiple == 0u){
                delete[] pointer;
            } else {
                pointer = NULL;
            }
        } else {
            delete[] pointer;
        }
    }
    isValid = false;
    isMultiple = false;
    indexMultiple = 0u;
}
void Data::DeleteMultiple(Data* dataArray_in, unsigned length_in){
    for(unsigned i = 0u; i < length_in; i++){
        dataArray_in[i].DeletePointer();
    }
}
Data::~Data(){
    DeletePointer();
    delete[] offsetPointer;
    delete[] lengthArray;
    delete[] names;
}