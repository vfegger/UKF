#include "../include/Parser.hpp"

Parser::Parser(unsigned length_in){
    length = length_in;
    fileArray = new std::fstream[length_in];
    count = 0u;
}

unsigned Parser::OpenFile(std::string name_in, std::string extension_in, unsigned index_in){
    std::cout << "Trying to open file with name: " << name_in << "\n";
    if(index_in == UINT_MAX_VALUE){
        if(count >= length){
            std::cout << "Error: Too many files are open at the moment. Use old indexes to reopen in the same place or close all files.";
            return UINT_MAX_VALUE;
        }
        fileArray[count] = std::fstream(name_in + extension_in);
        count++;
        return count-1;
    } else {
        if(index_in >= length){
            std::cout << "Error: Index is out of range.";
            return UINT_MAX_VALUE;
        }
        if(fileArray[index_in].is_open()){
            fileArray[index_in].close();
        }
        fileArray[index_in] = std::fstream(name_in + extension_in);
        return index_in;
    }
}

void Parser::ImportConfiguration(unsigned index_in, std::string& name_out, unsigned& length_out){
    char name_aux[100];
    fileArray[index_in].getline(name_aux,100,'\n');
    name_out = name_aux;
    fileArray[index_in] >> length_out;
}
void Parser::ImportData(unsigned index_in, unsigned length_in, double* data_out){
    for(unsigned i = 0u; i < length_in; i++){
        fileArray[index_in] >> data_out[i];
    }
}
void Parser::ImportParameter(unsigned index_in, unsigned length_in, void* paramater_out, unsigned sizeType_in){
    for(unsigned i = 0u; i < length_in * sizeType_in; i++){
        fileArray[index_in] >> ((char*)paramater_out)[i];
    }
}

void Parser::ExportConfiguration(unsigned index_in, std::string name_in, unsigned length_in){
    fileArray[index_in] << name_in << "\n";
    fileArray[index_in] << length_in << "\n";
}
void Parser::ExportData(unsigned index_in, unsigned length_in, double* data_in){
    for(unsigned i = 0u; i < length_in; i++){
        fileArray[index_in] << data_in[i];
    }
}
void Parser::ExportParameter(unsigned index_in, unsigned length_in, void* paramater_in, unsigned sizeType_in){
    for(unsigned i = 0u; i < length_in * sizeType_in; i++){
        fileArray[index_in] << ((char*)paramater_in)[i];
    }
}

Parser::~Parser(){
    length = 0;
    delete[] fileArray;
    count = 0u;
}