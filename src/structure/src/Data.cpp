#include "../include/Data.hpp"

Data::Data(){
    lengthElements = 0;
    names = NULL;
    lengthArray = NULL;
    offsetPointer = NULL;
    pointer = NULL;
    length = 0u;
    count = 0u;
    isValid = false;

    multipleIndex = 0u;
    multipleLenght = 0u;
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
    count = 0u;

    multipleIndex = 0u;
    multipleLenght = 0u;
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
    multipleIndex = 0u;
    multipleLenght = 0u;
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
    if(pointer != NULL && multipleIndex == 0u) {
        delete[] pointer;
    }
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
void Data::InstantiateMultiple(Data*& dataArray_out, const Data& data_in, unsigned length_in, bool resetValues){
    if(length_in == 0u){
        std::cout << "Error: Invalid size. It is expected at least 1.";
        return;
    }
    if(!data_in.GetValidation() && data_in.GetLength() == 0u){
        std::cout << "Error: Invalid data for copy.";
    }
    if(dataArray_out != NULL){
        delete[] dataArray_out;
    }
    dataArray_out = new(std::nothrow) Data[length_in];
    for(unsigned i = 0u; i < length_in; i++){
        dataArray_out[i].lengthArray = new unsigned[data_in.lengthElements];
        dataArray_out[i].names = new std::string[data_in.lengthElements];
        dataArray_out[i].offsetPointer = new double*[data_in.lengthElements];
        for(unsigned j = 0u; j < data_in.lengthElements; j++){
            dataArray_out[i].names[j] = data_in.names[j];
            dataArray_out[i].lengthArray[j] = data_in.lengthArray[j];
        }
    }

    unsigned length_aux = length_in * data_in.GetLength();
    double* pointer = new double[length_aux];
    unsigned offset_aux = 0u;
    for(unsigned i = 0u; i < length_in; i++){
        dataArray_out[i].length = data_in.GetLength();
        dataArray_out[i].pointer = pointer + offset_aux;
        for(unsigned j = 0u; j < data_in.lengthElements; j++){
            dataArray_out[i].offsetPointer[j] = pointer + offset_aux;
            offset_aux += dataArray_out[i].lengthArray[j];
        }
    }

    if(resetValues) {
        for(unsigned i = 0u; i < length_in; i++){
            for(unsigned j = 0u; j < data_in.GetLength(); j++){
                dataArray_out[i].pointer[j] = 0.0;
            }
            dataArray_out[i].isValid = true;
            dataArray_out[i].multipleIndex = i;
            dataArray_out[i].multipleLenght = length_in;
        }
    } else {
        for(unsigned i = 0u; i < length_in; i++){
            for(unsigned j = 0u; j < data_in.GetLength(); j++){
                dataArray_out[i].pointer[j] = data_in.pointer[j];
            }
            dataArray_out[i].isValid = true;
            dataArray_out[i].multipleIndex = i;
            dataArray_out[i].multipleLenght = length_in;
        }
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
unsigned Data::GetCapacity() const {
    return lengthElements;
}
double* Data::GetPointer() const {
    if(isValid == false){
        std::cout << "Error: Pointer is not initialized.";
        return NULL;
    }
    return pointer;
}
bool Data::GetValidation() const {
    return isValid;
}
unsigned Data::GetCount() const {
    return count;
}
unsigned Data::GetLength() const {
    if(isValid == false){
        std::cout << "Error: Length of pointer is not initialized.";
        return 0u;
    }
    return length;
}
unsigned Data::GetLength(unsigned index_in) const {
    if(index_in >= count){
        std::cout << "Error: Invalid index access.";
        return 0u;
    }
    return lengthArray[index_in];
}
std::string Data::GetNames(unsigned index_in) const {
    if(index_in >= count){
        std::cout << "Error: Invalid index access.";
        return 0u;
    }
    return names[index_in];
}
double*& Data::operator[](unsigned index){
    return offsetPointer[index];
}
Data::~Data(){
    if(multipleIndex == 0u) {
        delete[] pointer;
    }
    delete[] offsetPointer;
    delete[] lengthArray;
    delete[] names;
    
    lengthElements = 0;
    names = NULL;
    lengthArray = NULL;
    offsetPointer = NULL;
    pointer = NULL;
    length = 0u;
    count = 0u;
    isValid = false;

    multipleIndex = 0u;
    multipleLenght = 0u;
}