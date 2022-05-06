#ifndef PARSER_HEADER
#define PARSER_HEADER

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

#define UINT_MAX_VALUE 4294967295

enum ParserType {
    Char,
    UChar,
    SInt,
    SUInt,
    Int,
    UInt,
    LInt,
    LUInt,
    LLInt,
    LLUInt,
    Float,
    Double,
    LDouble
};

class Parser {
private:
    std::ofstream* fileArray;
    unsigned length;
    unsigned count;
public:
    Parser(unsigned length_in);
    unsigned OpenFile(std::string path_in, std::string name_in, std::string extension_in, std::ios::openmode mode_in, unsigned index = UINT_MAX_VALUE);
    std::ofstream& GetStream(unsigned index_in);
    void CloseFile(unsigned index_in);

    static void ConvertToBinary(std::string path_in, std::string path_out);
    static void ConvertToText(std::string path_in, std::string path_out);


    static void ImportConfiguration(std::ifstream& file_in, std::string& name_out, unsigned& length_out, ParserType& type_out);
    static void ExportConfiguration(std::ofstream& file_in, std::string name_in, unsigned length_in, ParserType type_in);
    
    template<typename T>
    static void ImportValues(std::ifstream& file_in, unsigned length_in, void*& values_out);
    template<typename T>
    static void ExportValues(std::ofstream& file_in, unsigned length_in, void* values_in);

    static void ImportValues(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out);
    static void ExportValues(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in);
    

    static void ImportConfigurationBinary(std::ifstream& file_in, std::string& name_out, unsigned& length_out, ParserType& type_out);
    static void ExportConfigurationBinary(std::ofstream& file_in, std::string name_in, unsigned length_in, ParserType type_in);
    
    template<typename T>
    static void ImportValuesBinary(std::ifstream& file_in, unsigned length_in, void*& values_out);
    template<typename T>
    static void ExportValuesBinary(std::ofstream& file_in, unsigned length_in, void* values_in);

    static void ImportValuesBinary(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out);
    static void ExportValuesBinary(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in);
    

    //static void NewValues(void* values_in, ParserType type_in);
    static void DeleteValues(void* values_in, ParserType type_in);

    ~Parser();
};

template<typename T>
void Parser::ImportValues(std::ifstream& file_in, unsigned length_in, void*& values_out){
    for(unsigned i = 0u; i < length_in; i++){
        file_in >> ((T*)values_out)[i];
    }
}

template<typename T>
void Parser::ExportValues(std::ofstream& file_in, unsigned length_in, void* values_in){
    for(unsigned i = 0u; i < length_in; i++){
        file_in << ((T*)values_in)[i] << "\n";
    }
}

template<typename T>
void Parser::ImportValuesBinary(std::ifstream& file_in, unsigned length_in, void*& values_out){
    file_in.read((char*)values_out,sizeof(T)*length_in/sizeof(char));
}

template<typename T>
void Parser::ExportValuesBinary(std::ofstream& file_in, unsigned length_in, void* values_in){
    file_in.write((char*)values_in,sizeof(T)*length_in/sizeof(char));
}

#endif