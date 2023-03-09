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

int RunCase(std::string &path_binary, std::string &extension_binary,
            unsigned Lr, unsigned Lth, unsigned Lz, unsigned Lt,
            double Sr, double Sth, double Sz, double St,
            double T0, double sT0, double sTm0, double Q0, double sQ0,
            double Amp, double r0, double mean, double sigma,
            double alpha, double beta, double kappa,
            PointerType type_in, PointerContext context_in)
{
    std::cout << std::cout.precision(3);
    Pointer<Parser> parser = MemoryHandler::AllocValue<Parser, unsigned>(4u, PointerType::CPU, PointerContext::CPU_Only);
    unsigned indexTimer;
    unsigned indexTemperature;
    unsigned indexTemperatureMeasured;
    unsigned indexHeatFlux;

    std::streampos positionTimer;
    std::streampos positionTemperature;
    std::streampos positionTemperatureMeasured;
    std::streampos positionHeatFlux;

    std::string name_timer = "Timer";
    std::string name_temperature = "Temperature";
    std::string name_temperature_measured = "Temperature_measured";
    std::string name_heatFlux = "HeatFlux";
    std::string name_type = (type_in == PointerType::CPU) ? "_CPU" : "_GPU";

    std::string name_timer_aux = name_timer + "R" + std::to_string(Lr) + "Th" + std::to_string(Lth) + "Z" + std::to_string(Lz) + "T" + std::to_string(Lt) + name_type;
    std::string name_temperature_aux = name_temperature + "R" + std::to_string(Lr) + "Th" + std::to_string(Lth) + "Z" + std::to_string(Lz) + "T" + std::to_string(Lt) + name_type;
    std::string name_temperature_measured_aux = name_temperature_measured + "R" + std::to_string(Lr) + "Th" + std::to_string(Lth) + "Z" + std::to_string(Lz) + "T" + std::to_string(Lt) + name_type;
    std::string name_heatFlux_aux = name_heatFlux + "R" + std::to_string(Lr) + "Th" + std::to_string(Lth) + "Z" + std::to_string(Lz) + "T" + std::to_string(Lt) + name_type;

    indexTimer = parser.pointer[0u].OpenFileOut(path_binary, name_timer_aux, extension_binary, std::ios::binary);
    indexTemperature = parser.pointer[0u].OpenFileOut(path_binary, name_temperature_aux, extension_binary, std::ios::binary);
    indexTemperatureMeasured = parser.pointer[0u].OpenFileOut(path_binary, name_temperature_measured_aux, extension_binary, std::ios::binary);
    indexHeatFlux = parser.pointer[0u].OpenFileOut(path_binary, name_heatFlux_aux, extension_binary, std::ios::binary);

    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTimer), "Timer", UKF_TIMER + 1, ParserType::Double, positionTimer);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTemperature), "Temperature", Lr * Lth * Lz, ParserType::Double, positionTemperature);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTemperatureMeasured), "Temperature Measured", HCRC_Measures, ParserType::Double, positionTemperatureMeasured);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexHeatFlux), "Heat Flux", Lth * Lz, ParserType::Double, positionHeatFlux);

    HFG_CRC generator(Lr, Lth, Lz, Lt, Sr, Sth, Sz, St, T0, Q0, Amp, r0, type_in, context_in);
    generator.Generate(mean, sigma, MemoryHandler::GetCuBLASHandle(0u), MemoryHandler::GetStream(0u));

    HFE_CRC problem(Lr, Lth, Lz, Lt, Sr, Sth, Sz, St, T0, sT0, sTm0, Q0, sQ0, Amp, r0, type_in, context_in);

    problem.UpdateMeasure(generator.GetTemperature(0u), type_in, context_in);

    UKF ukf(Pointer<UKFMemory>(problem.GetMemory().pointer, problem.GetMemory().type, problem.GetMemory().context), alpha, beta, kappa);

    Pointer<Timer> timer = MemoryHandler::AllocValue<Timer, unsigned>(UKF_TIMER + 1u, PointerType::CPU, PointerContext::CPU_Only);
    double *timer_pointer = timer.pointer[0u].GetValues();

    Pointer<double> temperature_out = problem.GetMemory().pointer[0u].GetState().pointer[0u].GetPointer();
    Pointer<double> heatFlux_out = Pointer<double>(temperature_out.pointer + Lr * Lth * Lz, temperature_out.type, temperature_out.context);
    Pointer<double> temperatureMeasured_out = problem.GetMemory().pointer[0u].GetMeasure().pointer[0u].GetPointer();

    Pointer<double> temperature_parser, heatFlux_parser, temperatureMeasured_parser;
    if (type_in == PointerType::GPU)
    {
        temperature_parser = MemoryHandler::Alloc<double>(Lr * Lth * Lz, PointerType::CPU, PointerContext::CPU_Only);
        heatFlux_parser = MemoryHandler::Alloc<double>(Lth * Lz, PointerType::CPU, PointerContext::CPU_Only);
        temperatureMeasured_parser = MemoryHandler::Alloc<double>(HCRC_Measures, PointerType::CPU, PointerContext::CPU_Only);
        MemoryHandler::Copy(temperatureMeasured_parser, temperatureMeasured_out, HCRC_Measures);
    }
    else
    {
        temperature_parser = temperature_out;
        heatFlux_parser = heatFlux_out;
        temperatureMeasured_parser = temperatureMeasured_out;
    }
    Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperatureMeasured), HCRC_Measures, ParserType::Double, temperatureMeasured_parser.pointer, positionTemperatureMeasured, 0u);
    for (unsigned i = 1u; i <= Lt; i++)
    {
        std::cout << "Iteration " << i << ":\n";
        ukf.Iterate(timer.pointer[0u]);
        problem.UpdateMeasure(generator.GetTemperature(i), type_in, context_in);

        if (type_in == PointerType::GPU)
        {
            MemoryHandler::Copy(temperature_parser, temperature_out, Lr * Lth * Lz);
            MemoryHandler::Copy(heatFlux_parser, heatFlux_out, Lth * Lz);
            MemoryHandler::Copy(temperatureMeasured_parser, temperatureMeasured_out, HCRC_Measures);
            cudaDeviceSynchronize();
        }

        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperature), Lr * Lth * Lz, ParserType::Double, temperature_parser.pointer, positionTemperature, i - 1u);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexHeatFlux), Lth * Lz, ParserType::Double, heatFlux_parser.pointer, positionHeatFlux, i - 1u);
        timer.pointer[0u].Save();
        timer.pointer[0u].SetValues();
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTimer), UKF_TIMER + 1, ParserType::Double, timer_pointer, positionTimer, i - 1u);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperatureMeasured), HCRC_Measures, ParserType::Double, temperatureMeasured_parser.pointer, positionTemperatureMeasured, i);
    }

    MemoryHandler::Free<Parser>(parser);
    MemoryHandler::Free<Timer>(timer);
    if (type_in == PointerType::GPU)
    {
        MemoryHandler::Free(temperature_parser);
        MemoryHandler::Free(heatFlux_parser);
        MemoryHandler::Free(temperatureMeasured_parser);
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

    if (argc == 1 + 4 + 2)
    {
        Lr = (unsigned)std::stoi(argv[1u]);
        Lth = (unsigned)std::stoi(argv[2u]);
        Lz = (unsigned)std::stoi(argv[3u]);
        Lt = (unsigned)std::stoi(argv[4u]);
        pointerType = (PointerType)std::stoi(argv[5u]);
        pointerContext = (PointerContext)std::stoi(argv[6u]);
    }
    else
    {
        std::cout << "Error: Wrong number of parameters: " << argc << "\n";
        return 1;
    }

    if(pointerType == PointerType::GPU){
        cudaDeviceReset();
        MemoryHandler::CreateGPUContext(1u,1u,1u);
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
    unsigned iteration;
    void *pointer = NULL;

    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexSize), name, length, type, iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexSize), length, type, pointer, 0u);
    S = (double *)pointer;

    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexHPParm), name, length, type, iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexHPParm), length, type, pointer, 0u);
    HPParm = (double *)pointer;

    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexUKFParm), name, length, type, iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexUKFParm), length, type, pointer, 0u);
    UKFParm = (double *)pointer;

    delete parser;

    unsigned it;

    it = 0u;
    double Sr = S[it++];
    double Sth = S[it++];
    double Sz = S[it++];
    double St = S[it++];

    it = 0u;
    double T0 = HPParm[it++];
    double sT0 = HPParm[it++];
    double sTm0 = HPParm[it++];
    double Q0 = HPParm[it++];
    double sQ0 = HPParm[it++];
    double Amp = HPParm[it++];
    double r0 = HPParm[it++];
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

    RunCase(path_binary_out, extension_binary,
            Lr, Lth, Lz, Lt,
            Sr, Sth, Sz, St,
            T0, sT0, sTm0, Q0, sQ0, Amp, r0, mean, sigma,
            alpha, beta, kappa,
            pointerType, pointerContext);

    Parser::ConvertToText(path_binary_out, path_text_out, extension_text);

    std::string name_type = (pointerType == PointerType::GPU) ? "_GPU" : "_CPU";
    std::string ok_name = path_text_out +
                          "R" + std::to_string(Lr) +
                          "Th" + std::to_string(Lth) +
                          "Z" + std::to_string(Lz) +
                          "T" + std::to_string(Lt) +
                          name_type + ".ok";
    std::ofstream ok_file(ok_name);
    ok_file.close();

    if(pointerType == PointerType::GPU){
        MemoryHandler::DestroyGPUContext();
        cudaDeviceReset();
    }

    std::cout << "\nEnd Execution\n";
    return 0;
}