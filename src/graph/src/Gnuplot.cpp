#include "Gnuplot.hpp"


void GnuplotParser::ImportConfigurationGnuplot(std::ifstream& file_in, std::string& name_out, unsigned& length_out, ParserType& type_out, unsigned& iteration_out){
    char aux[102];
    unsigned type;
    file_in.getline(aux,102);
    name_out = aux+2;
    if(name_out.size() > 0u && name_out[name_out.size()-1u] == '\r'){
        name_out.pop_back();
    }
    file_in >> length_out;
    file_in >> type;
    file_in >> iteration_out;
    type_out = (ParserType)type;
}

void GnuplotParser::ExportConfigurationGnuplot(std::ofstream& file_in, std::string name_in, unsigned length_in, ParserType type_in, std::streampos& iterationPosition_out){
    std::string iteration_string = std::to_string(0u) + "    ";
    file_in << "& " << name_in << "\n";
    file_in << "& " << length_in << "\t";
    file_in << (unsigned)type_in << "\t";
    iterationPosition_out = file_in.tellp();
    file_in << iteration_string << "\n";
}

void GnuplotParser::ImportIndexValues(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out, unsigned iteration_in){
    std::streampos position = file_in.tellg();
    for(unsigned i = 0u; i < iteration_in * length_in; i++){
        file_in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    switch (type_in)
    {
    case ParserType::Char:
        values_out = new char[length_in];
        ImportIndexValues<char>(file_in, length_in, values_out);
        break;
    case ParserType::UChar:
        values_out = new unsigned char[length_in];
        ImportIndexValues<unsigned char>(file_in, length_in, values_out);
        break;
    case ParserType::SInt:
        values_out = new short int[length_in];
        ImportIndexValues<short int>(file_in, length_in, values_out);
        break;
    case ParserType::SUInt:
        values_out = new short unsigned int[length_in];
        ImportIndexValues<short unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::Int:
        values_out = new int[length_in];
        ImportIndexValues<int>(file_in, length_in, values_out);
        break;
    case ParserType::UInt:
        values_out = new unsigned int[length_in];
        ImportIndexValues<unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::LInt:
        values_out = new long int[length_in];
        ImportIndexValues<long int>(file_in, length_in, values_out);
        break;
    case ParserType::LUInt:
        values_out = new long unsigned int[length_in];
        ImportIndexValues<long unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::LLInt:
        values_out = new long long int[length_in];
        ImportIndexValues<long long int>(file_in, length_in, values_out);
        break;
    case ParserType::LLUInt:
        values_out = new long long unsigned int[length_in];
        ImportIndexValues<long long unsigned int>(file_in, length_in, values_out);
        break;
    case ParserType::Float:
        values_out = new float[length_in];
        ImportIndexValues<float>(file_in, length_in, values_out);
        break;
    case ParserType::Double:
        values_out = new double[length_in];
        ImportIndexValues<double>(file_in, length_in, values_out);
        break;
    case ParserType::LDouble:
        values_out = new long double[length_in];
        ImportIndexValues<long double>(file_in, length_in, values_out);
        break;
    default:
        break;
    }
    file_in.seekg(position);
}

void GnuplotParser::ExportIndexValues(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in, std::streampos& iterationPosition_in, unsigned iteration_in){
    std::streampos position = file_in.tellp();
    file_in.seekp(iterationPosition_in);
    file_in << iteration_in + 1u;
    file_in.seekp(position);
    switch (type_in)
    {
    case ParserType::Char:
        ExportIndexValues<char>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::UChar:
        ExportIndexValues<unsigned char>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::SInt:
        ExportIndexValues<short int>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::SUInt:
        ExportIndexValues<short unsigned int>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::Int:
        ExportIndexValues<int>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::UInt:
        ExportIndexValues<unsigned int>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::LInt:
        ExportIndexValues<long int>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::LUInt:
        ExportIndexValues<long unsigned int>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::LLInt:
        ExportIndexValues<long long int>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::LLUInt:
        ExportIndexValues<long long unsigned int>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::Float:
        ExportIndexValues<float>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::Double:
        ExportIndexValues<double>(file_in, iteration_in, length_in, values_in);
        break;
    case ParserType::LDouble:
        ExportIndexValues<long double>(file_in, iteration_in, length_in, values_in);
        break;
    default:
        break;
    }
}


void GnuplotParser::ImportAllIndexValues(std::ifstream& file_in, unsigned length_in, ParserType type_in, void*& values_out, unsigned iteration_in){
    unsigned size = length_in * iteration_in;
    switch (type_in)
    {
    case ParserType::Char:
        values_out = new char[size];
        ImportIndexValues<char>(file_in, size, values_out);
        break;
    case ParserType::UChar:
        values_out = new unsigned char[size];
        ImportIndexValues<unsigned char>(file_in, size, values_out);
        break;
    case ParserType::SInt:
        values_out = new short int[size];
        ImportIndexValues<short int>(file_in, size, values_out);
        break;
    case ParserType::SUInt:
        values_out = new short unsigned int[size];
        ImportIndexValues<short unsigned int>(file_in, size, values_out);
        break;
    case ParserType::Int:
        values_out = new int[size];
        ImportIndexValues<int>(file_in, size, values_out);
        break;
    case ParserType::UInt:
        values_out = new unsigned int[size];
        ImportIndexValues<unsigned int>(file_in, size, values_out);
        break;
    case ParserType::LInt:
        values_out = new long int[size];
        ImportIndexValues<long int>(file_in, size, values_out);
        break;
    case ParserType::LUInt:
        values_out = new long unsigned int[size];
        ImportIndexValues<long unsigned int>(file_in, size, values_out);
        break;
    case ParserType::LLInt:
        values_out = new long long int[size];
        ImportIndexValues<long long int>(file_in, size, values_out);
        break;
    case ParserType::LLUInt:
        values_out = new long long unsigned int[size];
        ImportIndexValues<long long unsigned int>(file_in, size, values_out);
        break;
    case ParserType::Float:
        values_out = new float[size];
        ImportIndexValues<float>(file_in, size, values_out);
        break;
    case ParserType::Double:
        values_out = new double[size];
        ImportIndexValues<double>(file_in, size, values_out);
        break;
    case ParserType::LDouble:
        values_out = new long double[size];
        ImportIndexValues<long double>(file_in, size, values_out);
        break;
    default:
        break;
    }
}

void GnuplotParser::ExportAllIndexValues(std::ofstream& file_in, unsigned length_in, ParserType type_in, void* values_in, std::streampos& iterationPosition_in, unsigned iteration_in){
    std::streampos position = file_in.tellp();
    file_in.seekp(iterationPosition_in);
    file_in << iteration_in;
    file_in.seekp(position);
    unsigned size = length_in * iteration_in;
    switch (type_in)
    {
    case ParserType::Char:
        ExportIndexValues<char>(file_in, 0u, size, values_in);
        break;
    case ParserType::UChar:
        ExportIndexValues<unsigned char>(file_in, 0u, size, values_in);
        break;
    case ParserType::SInt:
        ExportIndexValues<short int>(file_in, 0u, size, values_in);
        break;
    case ParserType::SUInt:
        ExportIndexValues<short unsigned int>(file_in, 0u, size, values_in);
        break;
    case ParserType::Int:
        ExportIndexValues<int>(file_in, 0u, size, values_in);
        break;
    case ParserType::UInt:
        ExportIndexValues<unsigned int>(file_in, 0u, size, values_in);
        break;
    case ParserType::LInt:
        ExportIndexValues<long int>(file_in, 0u, size, values_in);
        break;
    case ParserType::LUInt:
        ExportIndexValues<long unsigned int>(file_in, 0u, size, values_in);
        break;
    case ParserType::LLInt:
        ExportIndexValues<long long int>(file_in, 0u, size, values_in);
        break;
    case ParserType::LLUInt:
        ExportIndexValues<long long unsigned int>(file_in, 0u, size, values_in);
        break;
    case ParserType::Float:
        ExportIndexValues<float>(file_in, 0u, size, values_in);
        break;
    case ParserType::Double:
        ExportIndexValues<double>(file_in, 0u, size, values_in);
        break;
    case ParserType::LDouble:
        ExportIndexValues<long double>(file_in, 0u, size, values_in);
        break;
    default:
        break;
    }
}

void GnuplotParser::ConvertToGnuplot(std::string path_in, std::string path_out, std::string extension_in, std::string extension_out){
    std::string name;
    unsigned length;
    ParserType type;
    void* values;
    unsigned sizeType;
    std::string name_in;
    std::string name_out;
    for(const auto & entry : std::filesystem::directory_iterator(path_in)){
        name_in = path_in + entry.path().stem().string() + extension_in;
        name_out = path_out + entry.path().stem().string() + extension_out;
        
        std::ifstream in(name_in);
        std::ofstream out(name_out, std::ios::trunc);

        std::streampos iterationPosition;
        unsigned iteration = 0u;
        
        ImportConfiguration(in,name,length,type,iteration);
        ImportAllValues(in,length,type,values,iteration);

        ExportConfigurationGnuplot(out,name,length,type,iterationPosition);
        ExportAllIndexValues(out,length,type,values,iterationPosition,iteration);

        DeleteValues(values,type);

        out.close();
        in.close();
    }
}

