#include <iostream>
#include <stdio.h>
#include "../include/Gnuplot.hpp"

int main(int argc, char **argv)
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

    std::string name_aux, name_Heat, name_Temp, name_Temp_Meas, name_Heat_Sim, name_ErrorHeat, name_ErrorTemp;

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
        remove((path_text_out + name_aux + extension_text_out).c_str());
        remove((path_text_out + "Error" + name_aux + extension_text_out).c_str());
    }
    std::cout << "End Concat Errors to Heat Flux and Temperature File\n";

    // External Treatment for Profile Data
    unsigned Li, Lj, Lk, Lt;
    unsigned index1, index2;
    if (argc == 1 + 4 + 2)
    {
        Li = (unsigned)std::stoi(argv[1u]);
        Lj = (unsigned)std::stoi(argv[2u]);
        Lk = (unsigned)std::stoi(argv[3u]);
        Lt = (unsigned)std::stoi(argv[4u]);
        index1 = (unsigned)std::stoi(argv[5u]);
        index2 = (unsigned)std::stoi(argv[6u]);
        std::cout << "Arguments: " << Li << " " << Lj << " " << Lk << " " << Lt << " " << index1 << " " << index2 << "\n";
    }
    else
    {
        std::cout << "Error: Wrong number of parameters for generating profile data: " << argc << "\n";
        return 1;
    }
    std::cout << "Generate Profile Data for Heat Flux and Temperature\n";
    unsigned L[4u] = {Li, Lj, Lk, Lt};
    for (const auto &entry : std::filesystem::directory_iterator(path_text_in))
    {
        if (entry.path().extension().string() != extension_text_in)
        {
            continue;
        }
        name_aux = entry.path().stem().string();
        name_Heat = "HeatFlux";
        name_Temp = "Temperature";
        name_ErrorHeat = "ErrorHeatFlux";
        name_ErrorTemp = "ErrorTemperature";
        name_Temp_Meas = "Temperature_measured";
        name_Heat_Sim = "SimulationHeatFlux";

        bool bHeat = name_aux.compare(0, name_Heat.size(), name_Heat) == 0;
        bool bTemp = name_aux.compare(0, name_Temp.size(), name_Temp) == 0;
        bool bErrorHeat = name_aux.compare(0, name_ErrorHeat.size(), name_ErrorHeat) == 0;
        bool bErrorTemp = name_aux.compare(0, name_ErrorTemp.size(), name_ErrorTemp) == 0;
        bool bTempMeas = name_aux.compare(0, name_Temp_Meas.size(), name_Temp_Meas) == 0;
        bool bHeatSim = name_aux.compare(0, name_Heat_Sim.size(), name_Heat_Sim) == 0;
        if (!bHeat && !bTemp && !bErrorHeat && !bErrorTemp && !bTempMeas && !bHeatSim)
        {
            continue;
        }
        if(bTempMeas && name_aux.find("S0",0) != std::string::npos){
            continue;
        }
        name_in = path_text_in + name_aux + extension_text_in;
        name_out = path_text_out + "Profile" + name_aux + extension_text_out;

        std::cout << "\t" << name_in << " >> " << name_out << "\n";
        std::ifstream in(name_in);
        std::ofstream out(name_out, std::ios::trunc);

        unsigned stride[4u] = {1u, Li, Li * Lj, Li * Lj * Lk};
        if (bHeat || bTempMeas || bHeatSim || bErrorHeat)
        {
            stride[index1] = 1u;
            stride[index2] = L[index1];
            stride[3u] = L[index1] * L[index2];
        }
        std::cout << "\tStride: " << stride[0u] << " " << stride[1u] << " " << stride[2u] << " " << stride[3u] << " " << index1 << " " << index2 << "\n";
        if (!in.is_open())
        {
            std::cout << "\tError: Failed to open file input file.\n";
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

        std::string name;
        unsigned length;
        ParserType type;
        void *values;

        GnuplotParser::ImportConfiguration(in, name, length, type, iteration);
        GnuplotParser::ImportAllValues(in, length, type, values, iteration);

        for (unsigned j = 0u; j < L[index2]; j++)
        {
            for (unsigned i = 0u; i < L[index1]; i++)
            {
                out << ((double *)values)[Lt * stride[3u] + j * stride[index2] + i * stride[index1]];
                if (i < L[index1] - 1u)
                {
                    out << " ";
                }
            }
            if (j < L[index2] - 1u)
            {
                out << "\n";
            }
        }

        GnuplotParser::DeleteValues(values, type);
        out.close();
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