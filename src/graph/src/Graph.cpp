#include <iostream>
#include "../include/Gnuplot.hpp"

int main()
{
    std::cout << "\nStart Execution\n\n";
    std::string path_dir = std::filesystem::current_path().string();
    std::string path_text_in = path_dir + "/graph/data/in/";
    std::string path_text_out = path_dir + "/graph/data/out/";

    std::string extension_text_in = ".dat";
    std::string extension_text_out = ".dat";

    std::cout << "Conversion to Gnuplot\n";

    GnuplotParser::ConvertToGnuplot(path_text_in, path_text_out, extension_text_in, extension_text_out);

    // External Treatment for Error data
    std::cout << "Concat Errors to Heat Flux and Temperature File\n";

    std::string name_aux, name_Heat, name_Temp, name_Temp_Meas;

    std::string name_in, name_Error_in;
    std::string name_out;
    for (const auto &entry : std::filesystem::directory_iterator(path_text_in))
    {
        if (entry.path().extension().string() != extension_text_in)
        {
            continue;
        }
        name_aux = entry.path().stem().string();
        name_Heat = "HeatFlux";
        name_Temp = "Temperature";
        name_Temp_Meas = "Temperature_measured";
        if (name_aux.compare(0, name_Heat.size(), name_Heat) != 0 && name_aux.compare(0, name_Temp.size(), name_Temp) != 0)
        {
            continue;
        }
        else if (name_aux.compare(0, name_Temp_Meas.size(), name_Temp_Meas) == 0)
        {
            continue;
        }
        name_in = path_text_in + name_aux + extension_text_in;
        name_Error_in = path_text_in + "Error" + name_aux + extension_text_in;
        name_out = path_text_out + "ValuesWithError" + name_aux + extension_text_out;

        std::cout << "\t" << name_in << " + " << name_Error_in << " >> " << name_out << "\n";
        std::ifstream in(name_in);
        std::ifstream in_Error(name_Error_in);
        std::ofstream out(name_out, std::ios::trunc);

        if (!in.is_open())
        {
            std::cout << "\tError: Failed to open file input file.\n";
            continue;
        }
        if (!in_Error.is_open())
        {
            std::cout << "\tError: Failed to open file input error file.\n";
            continue;
        }
        if (!out.is_open())
        {
            std::cout << "\tError: Failed to open file output file.\n";
            continue;
        }

        std::streampos iterationPosition, iterationPosition_Error;
        unsigned iteration = 0u;
        unsigned iteration_Error = 0u;

        std::string name, name_Error;
        unsigned length, length_Error;
        ParserType type, type_Error;
        void *values, *values_Error;
        void **values_out;

        GnuplotParser::ImportConfiguration(in, name, length, type, iteration);
        GnuplotParser::ImportAllValues(in, length, type, values, iteration);

        GnuplotParser::ImportConfiguration(in_Error, name_Error, length_Error, type_Error, iteration_Error);
        GnuplotParser::ImportAllValues(in_Error, length_Error, type_Error, values_Error, iteration_Error);

        values_out = new void *[2u];
        values_out[0u] = values;
        values_out[1u] = values_Error;

        GnuplotParser::ExportConfigurationGnuplot(out, name + " with Error", length, type, iterationPosition);
        GnuplotParser::ExportAllIndexMultipleValues(out, length, 2u, type, values_out, iterationPosition, iteration);

        GnuplotParser::DeleteValues(values_Error, type);
        GnuplotParser::DeleteValues(values, type);
        delete[] values_out;

        out.close();
        in_Error.close();
        in.close();
    }
    std::cout << "End Concat Errors to Heat Flux and Temperature File\n";
    /*
    std::cout << "Binary Covariance Conversion\n";
    std::string name_covariance = "Covariance";
    for (const auto &entry : std::filesystem::directory_iterator(path_text_in))
    {
        name_aux = entry.path().stem().string();
        if (name_aux.compare(0, name_covariance.size(), name_covariance) != 0)
        {
            continue;
        }
        name_in = path_text_in + name_aux + extension_text_in;
        name_out = path_text_out + "Binary" + name_aux + extension_text_out;

        std::cout << "\t" << name_in << " + " << name_Error_in << " >> " << name_out << "\n";

        std::ifstream in(name_in);
        std::ofstream out(name_out, std::ios::trunc | std::ios::binary);

        std::string name;
        unsigned length;
        ParserType type;
        void *values;
        unsigned iteration = 0u;

        GnuplotParser::ImportConfiguration(in, name, length, type, iteration);
        GnuplotParser::ImportAllValues(in, length, type, values, iteration);

        unsigned size = sizeof(int) + sizeof(double);
        void *memory = malloc(size * length * iteration / sizeof(char));
        for (int i = 0u; i < length * iteration; i++)
        {
            *((int *)(((char *)memory) + i * size / sizeof(char))) = i;
            *((double *)(((char *)memory) + (i * size / sizeof(char)) + sizeof(int))) = ((double*)values)[i];
        }
        out.write((char *)memory, size * length * iteration / sizeof(char));

        free(memory);
        out.close();
        in.close();
    }
    std::cout << "End Binary Covariance Conversion\n";
    */
    std::cout << "\nEnd Execution\n";
    return 0;
}