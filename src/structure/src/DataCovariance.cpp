#include "../include/DataCovariance.hpp"

DataCovariance::DataCovariance(){
    pointer = NULL;
    length = 0u;
    offsetPointer = NULL;
    lengthArray = NULL;
    offsetArray = NULL;
    lengthElements = 0u;
    names = NULL;
    isValid = false;
    count = 0u;
}
DataCovariance::DataCovariance(unsigned lengthElements_in){
    pointer = NULL;
    length = 0u;
    lengthElements = lengthElements_in;
    names = new(std::nothrow) std::string[lengthElements];
    lengthArray = new(std::nothrow) unsigned[lengthElements];
    offsetPointer = new(std::nothrow) double*[lengthElements];
    offsetArray = new(std::nothrow) unsigned[lengthElements];
    if(offsetPointer == NULL || lengthArray == NULL || offsetArray == NULL || names == NULL){
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    isValid = false;
    count = 0u;
}
DataCovariance::DataCovariance(const Data& data_in){
    pointer = NULL;
    length = 0u;
    lengthElements = data_in.GetCapacity();
    names = new(std::nothrow) std::string[lengthElements];
    lengthArray = new(std::nothrow) unsigned[lengthElements];
    offsetPointer = new(std::nothrow) double*[lengthElements];
    offsetArray = new(std::nothrow) unsigned[lengthElements];
    if(offsetPointer == NULL || lengthArray == NULL || offsetArray == NULL || names == NULL){
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    count = data_in.GetCount();
    unsigned offset_aux = 0u;
    for(unsigned i = 0u; i < count; i++){
        offsetPointer[i] = NULL;
        names[i] = data_in.GetNames(i);
        lengthArray[i] = data_in.GetLength(i);
        offsetArray[i] = offset_aux;
        offset_aux += lengthArray[i];
    }
    for(unsigned i = count; i < lengthElements; i++){
        offsetPointer[i] = NULL;
        names[i] = "";
        lengthArray[i] = 0u;
        offsetArray[i] = offset_aux;
        offset_aux += lengthArray[i];
    }
    isValid = data_in.GetValidation();
    if(isValid){
        length = data_in.GetLength();
        pointer = new(std::nothrow) double[length*length];
        if(pointer == NULL){
            std::cout << "Error: Initialization wasn't successful.\n";
            return;
        }
        for(unsigned i = 0u; i < count; i++){
            offsetPointer[i] = pointer + offsetArray[i] * (length + 1u);
        }
        for(unsigned i = 0u; i < length*length; i++)
        {
            pointer[i] = 0u;
        }
    }
}
DataCovariance::DataCovariance(const DataCovariance& dataCovariance_in){
    pointer = NULL;
    length = 0u;
    lengthElements = dataCovariance_in.lengthElements;
    offsetPointer = new(std::nothrow) double*[lengthElements];
    lengthArray = new(std::nothrow) unsigned[lengthElements];
    offsetArray = new(std::nothrow) unsigned[lengthElements];
    names = new(std::nothrow) std::string[lengthElements];
    if(offsetPointer == NULL || lengthArray == NULL || offsetArray == NULL || names == NULL){
        std::cout << "Error: Data covariance could not alloc all needed memory.\n";
    }
    count = dataCovariance_in.count;
    for(unsigned i = 0u; i < lengthElements; i++){
        offsetPointer[i] = NULL;
        names[i] = dataCovariance_in.names[i];
        lengthArray[i] = dataCovariance_in.lengthArray[i];
        offsetArray[i] = dataCovariance_in.offsetArray[i];
    }
    isValid = dataCovariance_in.isValid;
    if(isValid){
        length = dataCovariance_in.length;
        pointer = new(std::nothrow) double[length*length];
        if(pointer == NULL){
            std::cout << "Error: Initialization wasn't successful.\n";
            return;
        }
        for(unsigned i = 0u; i < count; i++){
            offsetPointer[i] = pointer + offsetArray[i] * (length + 1u);
        }
        for(unsigned i = 0u; i < length*length; i++)
        {
            pointer[i] = dataCovariance_in.pointer[i];
        }
    }
}
unsigned DataCovariance::Add(std::string name_in, unsigned length_in){
    isValid = false;
    if(count >= lengthElements){
        std::cout << "Error: Added element is over the limit.\n";
        return lengthElements;
    }
    names[count] = name_in;
    lengthArray[count] = length_in;
    offsetArray[count] = offsetArray[count-1]+length_in;
    count++;
    return count-1;
}
void DataCovariance::Add(unsigned* indexes, std::string* names_in, unsigned* lengthArray_in, unsigned lengthElements_in){
    unsigned index = 0u;
    for(unsigned i = 0; i < lengthElements_in; i++){
        index = Add(names_in[i], lengthArray_in[i]);
        if(indexes != NULL){
            indexes[i] = index;
        }
    }
}
void DataCovariance::Initialize(){
    if(pointer != NULL) {
        delete[] pointer;
    }
    length = 0u;
    for(unsigned i = 0u; i < count; i++){
        length += lengthArray[i];
    }
    pointer = new(std::nothrow) double[length*length];
    if(pointer == NULL){
        std::cout << "Error: Initialization wasn't successful.\n";
        return;
    }
    for(unsigned i = 0u; i < length; i++)
    {
        pointer[i] = 0u;
    }
    for(unsigned i = 0u; i < count; i++){
        offsetPointer[i] = pointer + offsetArray[count] * (length + 1u);
    }
    isValid = true;
}
void DataCovariance::LoadData(unsigned index_in, double* array_in, unsigned length_in, DataCovarianceMode mode_in){
    if(isValid == false){
        std::cout << "Error: Load while structure is not initialized.\n";
        return;
    }
    if(index_in >= count){
        std::cout << "Error: Index is out of range.\n";
        return;
    }
    switch (mode_in)
    {
    case DataCovarianceMode::Natural :
        unsigned ii, jj;
        for(unsigned j = 0u; j < length_in; j++){
            for(unsigned i = 0; i < length_in; i++){
                offsetPointer[index_in][j*length+i] = array_in[j*length_in+i];
            }
        }
        break;
    case DataCovarianceMode::Compact :
        for(unsigned i = 0; i < length_in; i++){
            offsetPointer[index_in][i*length+i] = array_in[i];
        }
        break;
    case DataCovarianceMode::Complete :
        std::cout << "Warning: This mode overwrites the whole covariance matrix.\n";
        if(length != length_in){
            std::cout << "Error: The dimensions of the covariance matrix and the covariance matrix input do not match.\n";
        }
        for(unsigned j = 0u; j < length_in; j++){
            for(unsigned i = 0u; i < length_in; i++){
                offsetPointer[index_in][j*length+i] = array_in[j*length_in+i];
            }
        }
        break;
    default:
        std::cout << "Error: Covariance mode is not implemented.\n";
        break;
    }
}
void DataCovariance::LoadData(unsigned* indexes_in, double** array_in, unsigned* lengthArray_in, DataCovarianceMode* modeArray_in, unsigned lengthElements_in){
    for(unsigned i = 0u; i < lengthElements_in; i++){
        LoadData(indexes_in[i], array_in[i], lengthArray_in[i],modeArray_in[i]);
    }
}
unsigned DataCovariance::GetCapacity() const {
    return lengthElements;
}
double* DataCovariance::GetPointer() const {
    if(isValid == false){
        std::cout << "Error: Pointer is not initialized.\n";
        return NULL;
    }
    return pointer;
}
bool DataCovariance::GetValidation() const {
    return isValid;
}
unsigned DataCovariance::GetCount() const {
    return count;
}
unsigned DataCovariance::GetLength() const {
    if(isValid == false){
        std::cout << "Error: Length of pointer is not initialized.\n";
        return 0u;
    }
    return length;
}
unsigned DataCovariance::GetLength(unsigned index_in) const {
    if(index_in >= count){
        std::cout << "Error: Invalid index access.\n";
        return 0u;
    }
    return lengthArray[index_in];
}
unsigned DataCovariance::GetOffset(unsigned index_in) const {
    if(index_in >= count){
        std::cout << "Error: Invalid index access.\n";
        return 0u;
    }
    return offsetArray[index_in];
}
std::string DataCovariance::GetNames(unsigned index_in) const {
    if(index_in >= count){
        std::cout << "Error: Invalid index access.\n";
        return "";
    }
    return names[index_in];
}
double*& DataCovariance::operator[](unsigned index){
    return offsetPointer[index];
}
DataCovariance::~DataCovariance(){
    delete[] pointer;
    delete[] offsetPointer;
    delete[] lengthArray;
    delete[] offsetArray;
    delete[] names;
    
    lengthElements = 0;
    names = NULL;
    lengthArray = NULL;
    offsetPointer = NULL;
    pointer = NULL;
    length = 0u;
    count = 0u;
    isValid = false;
}
