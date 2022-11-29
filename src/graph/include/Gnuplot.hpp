#ifndef GNUPLOT_HEADER
#define GNUPLOT_HEADER

#include <iostream>
#include "../../parser/include/Parser.hpp"

class GnuplotParser : public Parser
{
private:
public:
    static void ImportConfigurationGnuplot(std::ifstream &file_in, std::string &name_out, unsigned &length_out, ParserType &type_out, unsigned &iteration_out);
    static void ExportConfigurationGnuplot(std::ofstream &file_in, std::string name_in, unsigned length_in, ParserType type_in, std::streampos &iterationPosition_out);

    static void ImportIndexValues(std::ifstream &file_in, unsigned length_in, ParserType type_in, void *&values_out, unsigned iteration_in = 0u);
    static void ExportIndexValues(std::ofstream &file_in, unsigned length_in, ParserType type_in, void *values_in, std::streampos &iterationPosition_in, unsigned iteration_in = 0u);

    template <typename T>
    static void ImportIndexValues(std::ifstream &file_in, unsigned length_in, void *&values_out);
    template <typename T>
    static void ExportIndexValues(std::ofstream &file_in, unsigned iteration_in, unsigned length_in, void *values_in);

    static void ImportAllIndexValues(std::ifstream &file_in, unsigned length_in, ParserType type_in, void *&values_out, unsigned iteration_in = 1u);
    static void ExportAllIndexValues(std::ofstream &file_in, unsigned length_in, ParserType type_in, void *values_in, std::streampos &iterationPosition_in, unsigned iteration_in = 1u);

    static void ConvertToGnuplot(std::string path_in, std::string path_out, std::string extension_in, std::string extension_out);
};

template <typename T>
void GnuplotParser::ImportIndexValues(std::ifstream &file_in, unsigned length_in, void *&values_out)
{
    unsigned index;
    for (unsigned i = 0u; i < length_in; i++)
    {
        file_in >> index;
        file_in >> ((T *)values_out)[i];
    }
}

template <typename T>
void GnuplotParser::ExportIndexValues(std::ofstream &file_in, unsigned iteration_in, unsigned length_in, void *values_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        file_in << iteration_in * length_in + i << "\t" << ((T *)values_in)[i] << "\n";
    }
}

#endif