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
    std::ifstream* fileArray_In;
    unsigned length_In;
    unsigned count_In;

    std::ofstream* fileArray_Out;
    unsigned length_Out;
    unsigned count_Out;
public:
    Parser(unsigned length_in);
    unsigned OpenFileIn(std::string path_in, std::string name_in, std::string extension_in, std::ios::openmode mode_in, unsigned index = UINT_MAX_VALUE);
    unsigned OpenFileOut(std::string path_in, std::string name_in, std::string extension_in, std::ios::openmode mode_in, unsigned index = UINT_MAX_VALUE);
    std::ifstream& GetStreamIn(unsigned index_in);
    std::ofstream& GetStreamOut(unsigned index_in);
    void CloseFileIn(unsigned index_in);
    void CloseFileOut(unsigned index_in);
    void CloseAllFileIn();
    void CloseAllFileOut();


    static void ConvertToBinary(std::string path_in, std::string path_out, std::string extension_in);
    static void ConvertToText(std::string path_in, std::string path_out, std::string extension_in);


    static void ImportConfiguration(std::ifstream& file_in, std::string& name_out, unsigned& length_out, ParserType& type_out, unsigned& iteration_out);
    static void ExportConfiguration(std::ofstream& file_in, std::string name_in, unsigned length_in, ParserType type_in, std::streampos& iterationPosition_out);
    
    template<typename T>
    static void ImportValues(std::ifstream& file_in, unsigned length_in, void*& values_out);
    template<typename T>
    static void ExportValues(std::ofstream& file_in, unsigned length_in, void* values_in);

    static void ImportValues(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out, unsigned iteration_in = 0u);
    static void ExportValues(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in, std::streampos& iterationPosition_in, unsigned iteration_in = 0u);
    
    static void ImportAllValues(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out, unsigned iteration_in = 1u);
    static void ExportAllValues(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in, std::streampos& iterationPosition_in, unsigned iteration_in = 1u);
    

    static void ImportConfigurationBinary(std::ifstream& file_in, std::string& name_out, unsigned& length_out, ParserType& type_out, unsigned& iteration_out);
    static void ExportConfigurationBinary(std::ofstream& file_in, std::string name_in, unsigned length_in, ParserType type_in, std::streampos& iterationPosition_out);
    
    template<typename T>
    static void ImportValuesBinary(std::ifstream& file_in, unsigned length_in, void*& values_out);
    template<typename T>
    static void ExportValuesBinary(std::ofstream& file_in, unsigned length_in, void* values_in);

    static void ImportValuesBinary(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out, unsigned iteration_in = 0u);
    static void ExportValuesBinary(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in, std::streampos& iterationPosition_in, unsigned iteration_in = 0u);
    
    static void ImportAllValuesBinary(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out, unsigned iteration_in = 1u);
    static void ExportAllValuesBinary(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in, std::streampos& iterationPosition_in, unsigned iteration_in = 1u);
    

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