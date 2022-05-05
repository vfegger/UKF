#ifndef PARSER_HEADER
#define PARSER_HEADER

#include <iostream>
#include <fstream>
#include <string>

#define UINT_MAX_VALUE 4294967295

class Parser {
private:
    std::fstream* fileArray;
    unsigned length;
    unsigned count;
public:
    Parser(unsigned length_in);

    unsigned OpenFile(std::string path_in, std::string name_in, std::string extension_in, std::ios::openmode mode_in, unsigned index = UINT_MAX_VALUE);
    void ImportConfiguration(unsigned index_in, std::string& name_out, unsigned& length_out);
    void ImportData(unsigned index_in, unsigned length_in, double* data_out);
    void ImportParameter(unsigned index_in, unsigned length_in, void* parameter_out, unsigned sizeType_in);
    template<typename T>
    void ImportParameter(unsigned index_in, unsigned length_in, void* parameter_out);

    void ExportConfiguration(unsigned index_in, std::string name_in, unsigned length_in);
    void ExportData(unsigned index_in, unsigned length_in, double* data_in);
    void ExportParameter(unsigned index_in, unsigned length_in, void* parameter_in, unsigned sizeType_in);
    template<typename T>
    void ExportParameter(unsigned index_in, unsigned length_in, void* parameter_in);

    ~Parser();
};

template<typename T>
void Parser::ImportParameter(unsigned index_in, unsigned length_in, void* paramater_out){
    for(unsigned i = 0u; i < length_in; i++){
        fileArray[index_in] >> ((T*)paramater_out)[i];
    }
}

template<typename T>
void Parser::ExportParameter(unsigned index_in, unsigned length_in, void* parameter_in){
    for(unsigned i = 0u; i < length_in; i++){
        fileArray[index_in] << ((T*)parameter_in)[i];
    }
}

#endif