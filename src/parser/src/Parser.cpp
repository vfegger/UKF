#include "../include/Parser.hpp"

Parser::Parser(unsigned length_in){
    length = length_in;
    fileArray = new std::ofstream[length_in];
    count = 0u;
}

void Parser::ImportConfiguration(std::ifstream& file_in, std::string& name_out, unsigned& length_out, ParserType& type_out){
    char aux[100];
    unsigned type;
    file_in.getline(aux,100);
    name_out = aux;
    if(name_out.size() > 0u && name_out[name_out.size()-1u] == '\r'){
        name_out.pop_back();
    }
    file_in >> length_out;
    file_in >> type;
    type_out = (ParserType)type;
}

void Parser::ExportConfiguration(std::ofstream& file_in, std::string name_in, unsigned length_in, ParserType type_in){
    file_in << name_in << "\n";
    file_in << length_in << "\t";
    file_in << (unsigned)type_in << "\n";
}

void Parser::ImportValues(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out){
    switch (type_in)
    {
    case ParserType::Char:
        values_out = new char[length_in];
        ImportValues<char>(file_in, length_in, values_out);
        break;
    case ParserType::UChar:
        values_out = new unsigned char[length_in];
        ImportValues<unsigned char>(file_in, length_in, values_out);
        break;
    case ParserType::SInt:
        values_out = new short int[length_in];
        ImportValues<short int>(file_in, length_in, values_out);
        break;
    case ParserType::SUInt:
        values_out = new short unsigned int[length_in];
        ImportValues<short unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::Int:
        values_out = new int[length_in];
        ImportValues<int>(file_in, length_in, values_out);
        break;
    case ParserType::UInt:
        values_out = new unsigned int[length_in];
        ImportValues<unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::LInt:
        values_out = new long int[length_in];
        ImportValues<long int>(file_in, length_in, values_out);
        break;
    case ParserType::LUInt:
        values_out = new long unsigned int[length_in];
        ImportValues<long unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::LLInt:
        values_out = new long long int[length_in];
        ImportValues<long long int>(file_in, length_in, values_out);
        break;
    case ParserType::LLUInt:
        values_out = new long long unsigned int[length_in];
        ImportValues<long long unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::Float:
        values_out = new float[length_in];
        ImportValues<float>(file_in, length_in, values_out);
        break;
    case ParserType::Double:
        values_out = new double[length_in];
        ImportValues<double>(file_in, length_in, values_out);
        break;
    case ParserType::LDouble:
        values_out = new long double[length_in];
        ImportValues<long double>(file_in, length_in, values_out);
        break;
    default:
        break;
    }
}

void Parser::ExportValues(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in){
    switch (type_in)
    {
    case ParserType::Char:
        ExportValues<char>(file_in, length_in, values_in);
        break;
    case ParserType::UChar:
        ExportValues<unsigned char>(file_in, length_in, values_in);
        break;
    case ParserType::SInt:
        ExportValues<short int>(file_in, length_in, values_in);
        break;
    case ParserType::SUInt:
        ExportValues<short unsigned int>(file_in, length_in, values_in);
        break;
    case ParserType::Int:
        ExportValues<int>(file_in, length_in, values_in);
        break;
    case ParserType::UInt:
        ExportValues<unsigned int>(file_in, length_in, values_in);
        break;
    case ParserType::LInt:
        ExportValues<long int>(file_in, length_in, values_in);
        break;
    case ParserType::LUInt:
        ExportValues<long unsigned int>(file_in, length_in, values_in);
        break;
    case ParserType::LLInt:
        ExportValues<long long int>(file_in, length_in, values_in);
        break;
    case ParserType::LLUInt:
        ExportValues<long long unsigned int>(file_in, length_in, values_in);
        break;
    case ParserType::Float:
        ExportValues<float>(file_in, length_in, values_in);
        break;
    case ParserType::Double:
        ExportValues<double>(file_in, length_in, values_in);
        break;
    case ParserType::LDouble:
        ExportValues<long double>(file_in, length_in, values_in);
        break;
    default:
        break;
    }
}

void Parser::ImportConfigurationBinary(std::ifstream& file_in, std::string& name_out, unsigned& length_out, ParserType& type_out){
    unsigned type;
    unsigned size = 0u;
    file_in.read((char*)(&size),sizeof(unsigned));
    char aux[size];
    file_in.read(aux,sizeof(char)*size);
    file_in.read((char*)(&length_out),sizeof(unsigned));
    file_in.read((char*)(&(type)),sizeof(unsigned));
    name_out = aux;
    type_out = (ParserType)type;
}

void Parser::ExportConfigurationBinary(std::ofstream& file_in, std::string name_in, unsigned length_in, ParserType type_in){
    unsigned size = name_in.size();
    unsigned type = (unsigned)type_in;
    file_in.write((char*)(&size),sizeof(unsigned));
    file_in.write(name_in.c_str(),sizeof(char)*size);
    file_in.write((char*)(&length_in),sizeof(unsigned));
    file_in.write((char*)(&(type)),sizeof(unsigned));
}

void Parser::ImportValuesBinary(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out){
    switch (type_in)
    {
    case ParserType::Char:
        values_out = new char[length_in];
        ImportValuesBinary<char>(file_in, length_in, values_out);
        break;
    case ParserType::UChar:
        values_out = new unsigned char[length_in];
        ImportValuesBinary<unsigned char>(file_in, length_in, values_out);
        break;
    case ParserType::SInt:
        values_out = new short int[length_in];
        ImportValuesBinary<short int>(file_in, length_in, values_out);
        break;
    case ParserType::SUInt:
        values_out = new short unsigned int[length_in];
        ImportValuesBinary<short unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::Int:
        values_out = new int[length_in];
        ImportValuesBinary<int>(file_in, length_in, values_out);
        break;
    case ParserType::UInt:
        values_out = new unsigned int[length_in];
        ImportValuesBinary<unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::LInt:
        values_out = new long int[length_in];
        ImportValuesBinary<long int>(file_in, length_in, values_out);
        break;
    case ParserType::LUInt:
        values_out = new long unsigned int[length_in];
        ImportValuesBinary<long unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::LLInt:
        values_out = new long long int[length_in];
        ImportValuesBinary<long long int>(file_in, length_in, values_out);
        break;
    case ParserType::LLUInt:
        values_out = new long long unsigned int[length_in];
        ImportValuesBinary<long long unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::Float:
        values_out = new float[length_in];
        ImportValuesBinary<float>(file_in, length_in, values_out);
        break;
    case ParserType::Double:
        values_out = new double[length_in];
        ImportValuesBinary<double>(file_in, length_in, values_out);
        break;
    case ParserType::LDouble:
        values_out = new long double[length_in];
        ImportValuesBinary<long double>(file_in, length_in, values_out);
        break;
    default:
        break;
    }
}

void Parser::ExportValuesBinary(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in){
    switch (type_in)
    {
    case ParserType::Char:
        ExportValuesBinary<char>(file_in, length_in, values_in);
        break;
    case ParserType::UChar:
        ExportValuesBinary<unsigned char>(file_in, length_in, values_in);
        break;
    case ParserType::SInt:
        ExportValuesBinary<short int>(file_in, length_in, values_in);
        break;
    case ParserType::SUInt:
        ExportValuesBinary<short unsigned int>(file_in, length_in, values_in);
        break;
    case ParserType::Int:
        ExportValuesBinary<int>(file_in, length_in, values_in);
        break;
    case ParserType::UInt:
        ExportValuesBinary<unsigned int>(file_in, length_in, values_in);
        break;
    case ParserType::LInt:
        ExportValuesBinary<long int>(file_in, length_in, values_in);
        break;
    case ParserType::LUInt:
        ExportValuesBinary<long unsigned int>(file_in, length_in, values_in);
        break;
    case ParserType::LLInt:
        ExportValuesBinary<long long int>(file_in, length_in, values_in);
        break;
    case ParserType::LLUInt:
        ExportValuesBinary<long long unsigned int>(file_in, length_in, values_in);
        break;
    case ParserType::Float:
        ExportValuesBinary<float>(file_in, length_in, values_in);
        break;
    case ParserType::Double:
        ExportValuesBinary<double>(file_in, length_in, values_in);
        break;
    case ParserType::LDouble:
        ExportValuesBinary<long double>(file_in, length_in, values_in);
        break;
    default:
        break;
    }
}

void Parser::DeleteValues(void* values_in, ParserType type_in){
    switch (type_in)
    {
    case ParserType::Char:
        delete[] ((char*)values_in);
        break;
    case ParserType::UChar:
        delete[] ((unsigned char*)values_in);
        break;
    case ParserType::SInt:
        delete[] ((short int*)values_in);
        break;
    case ParserType::SUInt:
        delete[] ((short unsigned int*)values_in);
        break;
    case ParserType::Int:
        delete[] ((int*)values_in);
        break;
    case ParserType::UInt:
        delete[] ((unsigned int*)values_in);
        break;
    case ParserType::LInt:
        delete[] ((long int*)values_in);
        break;
    case ParserType::LUInt:
        delete[] ((long unsigned int*)values_in);
        break;
    case ParserType::LLInt:
        delete[] ((long long int*)values_in);
        break;
    case ParserType::LLUInt:
        delete[] ((long long unsigned int*)values_in);
        break;
    case ParserType::Float:
        delete[] ((float*)values_in);
        break;
    case ParserType::Double:
        delete[] ((double*)values_in);
        break;
    case ParserType::LDouble:
        delete[] ((long double*)values_in);
        break;
    default:
        break;
    }
}

void Parser::ConvertToBinary(std::string path_in, std::string path_out){
    std::string name;
    unsigned length;
    ParserType type;
    void* values;
    unsigned sizeType;
    std::string name_in;
    std::string name_out;
    for(const auto & entry : std::filesystem::directory_iterator(path_in)){
        name_in = path_in + entry.path().filename().string();
        name_out = path_out + entry.path().filename().string();
        
        std::ifstream in(name_in);
        std::ofstream out(name_out, std::ios::binary | std::ios::trunc);
        
        ImportConfiguration(in,name,length,type);
        ImportValues(in,length,type,values);

        ExportConfigurationBinary(out,name,length,type);
        ExportValuesBinary(out,length,type,values);

        DeleteValues(values,type);

        out.close();
        in.close();
    }
}

void Parser::ConvertToText(std::string path_in, std::string path_out){
    std::string name;
    unsigned length;
    ParserType type;
    void* values;
    unsigned sizeType;
    for(const auto & entry : std::filesystem::directory_iterator(path_in)){
        std::ifstream in(path_in + entry.path().filename().string(), std::ios::binary);
        std::ofstream out(path_out + entry.path().filename().string(), std::ios::trunc);
        
        ImportConfigurationBinary(in,name,length,type);
        ImportValuesBinary(in,length,type,values);

        ExportConfiguration(out,name,length,type);
        ExportValues(out,length,type,values);

        DeleteValues(values,type);

        out.close();
        in.close();
    }
}


unsigned Parser::OpenFile(std::string path_in, std::string name_in, std::string extension_in, std::ios::openmode mode_in, unsigned index_in){
    std::cout << "Trying to open file with name: " << name_in << "\n";
    if(index_in == UINT_MAX_VALUE){
        if(count >= length){
            std::cout << "Error: Too many files are open at the moment. Use old indexes to reopen in the same place or close all files.";
            return UINT_MAX_VALUE;
        }
        fileArray[count] = std::ofstream(path_in + name_in + extension_in, mode_in);
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
        fileArray[index_in] = std::ofstream(path_in + name_in + extension_in, mode_in);
        return index_in;
    }
}

std::ofstream& Parser::GetStream(unsigned index_in){
    return fileArray[index_in];
}

void Parser::CloseFile(unsigned index_in){
    fileArray[index_in].close();
}

Parser::~Parser(){
    for(unsigned i = 0u; i < length; i++){
        if(fileArray[i].is_open()){
            fileArray[i].close();
        }
    }
    length = 0;
    delete[] fileArray;
    count = 0u;
}