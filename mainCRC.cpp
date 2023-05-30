#include <iostream>
#include "src/structure/include/Data.hpp"
#include "src/structure/include/DataCovariance.hpp"
#include "src/structure/include/Parameter.hpp"
#include "src/parser/include/Parser.hpp"
#include "src/ukf/include/UKF.hpp"
#include "src/timer/include/Timer.hpp"
#include "src/hfe-crc/include/HeatFluxEstimation.hpp"
#include "src/hfe-crc/include/HeatFluxGenerator.hpp"
#include <stdlib.h>
#include <filesystem>

int RunCase(std::string &path_binary_in, std::string &path_binary_out, std::string &extension_binary,
            unsigned Lr, unsigned Lth, unsigned Lz, unsigned Lt,
            double Sr, double Sth, double Sz, double St,
            double T0, double sT0, double sTm0, double Q0, double sQ0, double Tamb0, double sTamb0,
            double Amp, double r0, double h, double mean, double sigma,
            double alpha, double beta, double kappa,
            unsigned iteration, unsigned caseType, unsigned projectCase,
            PointerType type_in, PointerContext context_in)
{
    std::cout << std::setprecision(3);

    Pointer<Parser> parser = MemoryHandler::AllocValue<Parser, unsigned>(5u, PointerType::CPU, PointerContext::CPU_Only);

    // Input Treatment
    void *pointer = NULL;
    std::string name;
    unsigned length;
    ParserType type;
    unsigned it;
    std::string name_temperature_measured_input = "Temp" + std::to_string(projectCase);
    unsigned indexTemperatureMeasuredInput = parser.pointer[0u].OpenFileIn(path_binary_in, name_temperature_measured_input, extension_binary, std::ios::binary);
    Parser::ImportConfigurationBinary(parser.pointer[0u].GetStreamIn(indexTemperatureMeasuredInput), name, length, type, it);
    if (length != HCRC_Measures_Total)
    {
        std::cout << "Error: Input data do not match the expected size.\n";
        return 1;
    }
    if (Lt * iteration >= it)
    {
        Lt = it / iteration;
    }
    St = Lt * iteration * 0.2;
    Parser::ImportAllValuesBinary(parser.pointer[0u].GetStreamIn(indexTemperatureMeasuredInput), HCRC_Measures_Total, ParserType::Double, pointer, (Lt + 1u) * iteration);

    // Input Choice
    Pointer<double> measures;
    unsigned measuresLength = 0u;
    if (caseType == 0u)
    {
        measuresLength = HCRC_Measures;
        measures = MemoryHandler::Alloc<double>(HCRC_Measures * (Lt + 1), PointerType::CPU, PointerContext::CPU_Only);
        for (unsigned i = 0u; i <= Lt; i++)
        {
            unsigned j;
            for (j = 0u; j < HCRC_Measures - 1u; j++)
            {
                measures.pointer[i * HCRC_Measures + j] = ((double *)pointer)[i * iteration * HCRC_Measures_Total + j];
            }
            measures.pointer[i * HCRC_Measures + j] = ((double *)pointer)[i * iteration * HCRC_Measures_Total + HCRC_Measures_Total - 1];
        }
    }
    else if (caseType == 1u)
    {
        measuresLength = Lth * Lz;
        HFG_CRC *generator = new HFG_CRC(Lr, Lth, Lz, Lt, Sr, Sth, Sz, St, T0, Q0, Tamb0, Amp, r0, h, type_in, context_in, iteration);
        measures = MemoryHandler::Alloc<double>(Lth * Lz * (Lt + 1), PointerType::CPU, PointerContext::CPU_Only);
        generator->Generate(mean, sigma, MemoryHandler::GetCuBLASHandle(0u), MemoryHandler::GetStream(0u));
        generator->GetCompleteTemperatureBoundary(measures, MemoryHandler::GetStream(0u));
        cudaDeviceSynchronize();
        delete generator;
    }
    else
    {
        std::cout << "Error: Case Type is not defined.\n";
    }
    // Output Treatment
    unsigned indexTimer;
    unsigned indexTemperature;
    unsigned indexHeatFlux;

    std::streampos positionTimer;
    std::streampos positionTemperature;
    std::streampos positionHeatFlux;

    std::string name_timer = "Timer";
    std::string name_temperature = "Temperature";
    std::string name_heatFlux = "HeatFlux";
    std::string name_type = (type_in == PointerType::CPU) ? "_CPU" : "_GPU";

    std::string name_case = "R" + std::to_string(Lr) + "Th" + std::to_string(Lth) + "Z" + std::to_string(Lz) + "T" + std::to_string(Lt) + "S" + std::to_string(caseType) + "C" + std::to_string(projectCase);

    std::string name_timer_aux = name_timer + name_case + name_type;
    std::string name_temperature_aux = name_temperature + name_case + name_type;
    std::string name_heatFlux_aux = name_heatFlux + name_case + name_type;

    indexTimer = parser.pointer[0u].OpenFileOut(path_binary_out, name_timer_aux, extension_binary, std::ios::binary);
    indexTemperature = parser.pointer[0u].OpenFileOut(path_binary_out, name_temperature_aux, extension_binary, std::ios::binary);
    indexHeatFlux = parser.pointer[0u].OpenFileOut(path_binary_out, name_heatFlux_aux, extension_binary, std::ios::binary);

    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTimer), "Timer", UKF_TIMER + 1, ParserType::Double, positionTimer);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTemperature), "Temperature", Lr * Lth * Lz, ParserType::Double, positionTemperature);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexHeatFlux), "Heat Flux", Lth * Lz, ParserType::Double, positionHeatFlux);

    // Problem Definition
    HFE_CRC problem(Lr, Lth, Lz, Lt, Sr, Sth, Sz, St, T0, sT0, sTm0, Q0, sQ0, Tamb0, sTamb0, Amp, r0, h, caseType, type_in, context_in, iteration);

    problem.UpdateMeasure(measures, type_in, context_in);

    // UKF Setup
    UKF ukf(Pointer<UKFMemory>(problem.GetMemory().pointer, problem.GetMemory().type, problem.GetMemory().context), alpha, beta, kappa);

    // Timer Setup
    Pointer<Timer> timer = MemoryHandler::AllocValue<Timer, unsigned>(UKF_TIMER + 1u, PointerType::CPU, PointerContext::CPU_Only);
    double *timer_pointer = timer.pointer[0u].GetValues();

    Pointer<double> temperature_out = problem.GetMemory().pointer[0u].GetState().pointer[0u].GetPointer();
    Pointer<double> heatFlux_out = Pointer<double>(temperature_out.pointer + Lr * Lth * Lz, temperature_out.type, temperature_out.context);

    Pointer<double> temperature_parser, heatFlux_parser, measures_aux;
    if (type_in == PointerType::GPU)
    {
        temperature_parser = MemoryHandler::Alloc<double>(Lr * Lth * Lz, PointerType::CPU, PointerContext::CPU_Only);
        heatFlux_parser = MemoryHandler::Alloc<double>(Lth * Lz, PointerType::CPU, PointerContext::CPU_Only);
    }
    else
    {
        temperature_parser = temperature_out;
        heatFlux_parser = heatFlux_out;
    }
    for (unsigned i = 1u; i <= Lt; i++)
    {
        std::cout << "Iteration " << i << ":\n";
        ukf.Iterate(timer.pointer[0u]);
        measures_aux = Pointer<double>(measures.pointer + i * measuresLength, PointerType::CPU, PointerContext::CPU_Only);
        problem.UpdateMeasure(measures_aux, type_in, context_in);

        if (type_in == PointerType::GPU)
        {
            MemoryHandler::Copy(temperature_parser, temperature_out, Lr * Lth * Lz);
            MemoryHandler::Copy(heatFlux_parser, heatFlux_out, Lth * Lz);
            cudaDeviceSynchronize();
        }

        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperature), Lr * Lth * Lz, ParserType::Double, temperature_parser.pointer, positionTemperature, i - 1u);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexHeatFlux), Lth * Lz, ParserType::Double, heatFlux_parser.pointer, positionHeatFlux, i - 1u);
        timer.pointer[0u].Save();
        timer.pointer[0u].SetValues();
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTimer), UKF_TIMER + 1, ParserType::Double, timer_pointer, positionTimer, i - 1u);
    }

    MemoryHandler::Free<Parser>(parser);
    MemoryHandler::Free<Timer>(timer);
    if (type_in == PointerType::GPU)
    {
        MemoryHandler::Free(temperature_parser);
        MemoryHandler::Free(heatFlux_parser);
    }
    return 0;
}

int main(int argc, char **argv)
{
    std::cout << "\nStart Execution\n\n";

    unsigned Lr;
    unsigned Lth;
    unsigned Lz;
    unsigned Lt;

    PointerType pointerType;
    PointerContext pointerContext;

    unsigned iteration;
    unsigned caseType;
    unsigned projectCase;

    if (argc == 1 + 4 + 2 + 3)
    {
        Lr = (unsigned)std::stoi(argv[1u]);
        Lth = (unsigned)std::stoi(argv[2u]);
        Lz = (unsigned)std::stoi(argv[3u]);
        Lt = (unsigned)std::stoi(argv[4u]);
        pointerType = (PointerType)std::stoi(argv[5u]);
        pointerContext = (PointerContext)std::stoi(argv[6u]);
        iteration = (unsigned)std::stoi(argv[7u]);
        caseType = (unsigned)std::stoi(argv[8u]);
        projectCase = (unsigned)std::stoi(argv[9u]);
    }
    else
    {
        std::cout << "Error: Wrong number of parameters: " << argc << "\n";
        return 1;
    }

    if (pointerType == PointerType::GPU)
    {
        cudaDeviceReset();
        MemoryHandler::CreateGPUContext(1u, 1u, 1u);
    }

    std::string path_dir = std::filesystem::current_path().string();
    std::string path_text_in = path_dir + "/data/text/in/";
    std::string path_binary_in = path_dir + "/data/binary/in/";

    std::string path_text_out = path_dir + "/data/text/out/";
    std::string path_binary_out = path_dir + "/data/binary/out/";

    std::string name_problem = "_CRC";

    std::string name_size = "ParameterSize" + name_problem;
    std::string name_heatProblem = "ParameterHeatProblem" + name_problem;
    std::string name_UKFParm = "ParameterUKF" + name_problem;

    std::string extension_text = ".dat";
    std::string extension_binary = ".bin";

    Parser::ConvertToBinary(path_text_in, path_binary_in, extension_binary);

    Parser *parser = new Parser(20u);

    unsigned indexSize = parser->OpenFileIn(path_binary_in, name_size, extension_binary, std::ios::binary);
    unsigned indexHPParm = parser->OpenFileIn(path_binary_in, name_heatProblem, extension_binary, std::ios::binary);
    unsigned indexUKFParm = parser->OpenFileIn(path_binary_in, name_UKFParm, extension_binary, std::ios::binary);

    double *S = NULL;
    double *HPParm = NULL;
    double *UKFParm = NULL;

    std::string name;
    unsigned length;
    ParserType type;
    unsigned iteration_file;
    void *pointer = NULL;

    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexSize), name, length, type, iteration_file);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexSize), length, type, pointer, 0u);
    S = (double *)pointer;

    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexHPParm), name, length, type, iteration_file);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexHPParm), length, type, pointer, 0u);
    HPParm = (double *)pointer;

    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexUKFParm), name, length, type, iteration_file);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexUKFParm), length, type, pointer, 0u);
    UKFParm = (double *)pointer;

    delete parser;

    unsigned it;

    it = 0u;
    double Sr = S[it++];
    double Sth = std::acos(-1.0);
    double Sz = S[it++];
    double St = 0.0;

    it = 0u;
    double T0 = HPParm[it++];
    double sT0 = HPParm[it++];
    double sTm0 = HPParm[it++];
    double Q0 = HPParm[it++];
    double sQ0 = HPParm[it++];
    double Tamb0 = HPParm[it++];
    double sTamb0 = HPParm[it++];
    double Amp = HPParm[it++];
    double r0 = HPParm[it++];
    double h = HPParm[it++];
    double mean = HPParm[it++];
    double sigma = HPParm[it++];

    it = 0u;
    double alpha = UKFParm[it++];
    double beta = UKFParm[it++];
    double kappa = UKFParm[it++];

    delete[] S;
    delete[] HPParm;
    delete[] UKFParm;

    std::cout << "\nRunning case with grid: (" << Lr << "," << Lth << "," << Lz << "," << Lt << ")\n\n";

    RunCase(path_binary_in, path_binary_out, extension_binary,
            Lr, Lth, Lz, Lt,
            Sr, Sth, Sz, St,
            T0, sT0, sTm0, Q0, sQ0, Tamb0, sTamb0,
            Amp, r0, h, mean, sigma,
            alpha, beta, kappa,
            iteration, caseType, projectCase,
            pointerType, pointerContext);

    Parser::ConvertToText(path_binary_out, path_text_out, extension_text);

    std::string name_type = (pointerType == PointerType::GPU) ? "_GPU" : "_CPU";
    std::string ok_name = path_text_out +
                          "R" + std::to_string(Lr) +
                          "Th" + std::to_string(Lth) +
                          "Z" + std::to_string(Lz) +
                          "T" + std::to_string(Lt) +
                          "S" + std::to_string(caseType) +
                          "C" + std::to_string(projectCase) +
                          name_type + ".ok";
    std::ofstream ok_file(ok_name);
    ok_file.close();

    if (pointerType == PointerType::GPU)
    {
        MemoryHandler::DestroyGPUContext();
        cudaDeviceReset();
    }

    std::cout << "\nEnd Execution\n";
    return 0;
}