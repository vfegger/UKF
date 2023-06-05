#include <iostream>
#include "src/structure/include/Data.hpp"
#include "src/structure/include/DataCovariance.hpp"
#include "src/structure/include/Parameter.hpp"
#include "src/parser/include/Parser.hpp"
#include "src/ukf/include/UKF.hpp"
#include "src/timer/include/Timer.hpp"
#include "src/hfe/include/HeatFluxEstimation.hpp"
#include "src/hfe/include/HeatFluxGenerator.hpp"
#include <stdlib.h>
#include <filesystem>

int RunCase(std::string &path_binary, std::string &extension_binary,
            unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt,
            double Sx, double Sy, double Sz, double St,
            double T0, double sT0, double sTm0, double Q0, double sQ0, double Amp, double mean, double sigma,
            double alpha, double beta, double kappa,
            PointerType type_in, PointerContext context_in)
{
    std::cout << std::setprecision(3);

    unsigned Lxy = Lx * Ly;
    unsigned Lxyz = Lx * Ly * Lz;
    unsigned L = Lxyz + Lxy;
    
    // Parser Initialization
    Pointer<Parser> parser = MemoryHandler::AllocValue<Parser, unsigned>(7u, PointerType::CPU, PointerContext::CPU_Only);

    // Output Setup
    unsigned indexTimer;
    unsigned indexTemperature;
    unsigned indexTemperatureMeasured;
    unsigned indexHeatFlux;
    unsigned indexTemperatureError;
    unsigned indexHeatFluxError;
    unsigned indexCovariance;

    std::streampos positionTimer;
    std::streampos positionTemperature;
    std::streampos positionTemperatureMeasured;
    std::streampos positionHeatFlux;
    std::streampos positionTemperatureError;
    std::streampos positionHeatFluxError;
    std::streampos positionCovariance;

    std::string name_timer = "Timer";
    std::string name_temperature = "Temperature";
    std::string name_temperature_measured = "Temperature_measured";
    std::string name_heatFlux = "HeatFlux";
    std::string name_temperatureError = "ErrorTemperature";
    std::string name_heatFluxError = "ErrorHeatFlux";
    std::string name_covariance = "Covariance";
    std::string name_type = (type_in == PointerType::CPU) ? "_CPU" : "_GPU";
    std::string name_case = "X" + std::to_string(Lx) + "Y" + std::to_string(Ly) + "Z" + std::to_string(Lz) + "T" + std::to_string(Lt);

    std::string name_timer_aux = name_timer + name_case + name_type;
    std::string name_temperature_aux = name_temperature + name_case + name_type;
    std::string name_temperature_measured_aux = name_temperature_measured + name_case + name_type;
    std::string name_heatFlux_aux = name_heatFlux + name_case + name_type;
    std::string name_temperatureError_aux = name_temperatureError + name_case + name_type;
    std::string name_heatFluxError_aux = name_heatFluxError + name_case + name_type;
    std::string name_covariance_aux = name_covariance + name_case + name_type;

    indexTimer = parser.pointer[0u].OpenFileOut(path_binary, name_timer_aux, extension_binary, std::ios::binary);
    indexTemperature = parser.pointer[0u].OpenFileOut(path_binary, name_temperature_aux, extension_binary, std::ios::binary);
    indexTemperatureMeasured = parser.pointer[0u].OpenFileOut(path_binary, name_temperature_measured_aux, extension_binary, std::ios::binary);
    indexHeatFlux = parser.pointer[0u].OpenFileOut(path_binary, name_heatFlux_aux, extension_binary, std::ios::binary);
    indexTemperatureError = parser.pointer[0u].OpenFileOut(path_binary, name_temperatureError_aux, extension_binary, std::ios::binary);
    indexHeatFluxError = parser.pointer[0u].OpenFileOut(path_binary, name_heatFluxError_aux, extension_binary, std::ios::binary);
    indexCovariance = parser.pointer[0u].OpenFileOut(path_binary, name_covariance_aux, extension_binary, std::ios::binary);

    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTimer), "Timer", UKF_TIMER + 1, ParserType::Double, positionTimer);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTemperature), "Temperature", Lxyz, ParserType::Double, positionTemperature);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTemperatureMeasured), "Temperature Measured", Lxy, ParserType::Double, positionTemperatureMeasured);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexHeatFlux), "Heat Flux", Lxy, ParserType::Double, positionHeatFlux);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTemperatureError), "Temperature Error", Lxyz, ParserType::Double, positionTemperatureError);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexHeatFluxError), "Heat Flux Error", Lxy, ParserType::Double, positionHeatFluxError);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexCovariance), "Covariance", L * L, ParserType::Double, positionCovariance);

    // Input Setup
    HeatFluxGenerator generator(Lx, Ly, Lz, Lt, Sx, Sy, Sz, St, T0, Q0, Amp, type_in, context_in);
    generator.Generate(mean, sigma, MemoryHandler::GetCuBLASHandle(0u), MemoryHandler::GetStream(0u));

    HeatFluxEstimation problem(Lx, Ly, Lz, Lt, Sx, Sy, Sz, St, T0, sT0, sTm0, Q0, sQ0, Amp, type_in, context_in);

    problem.UpdateMeasure(generator.GetTemperature(0u), Lx, Ly, type_in, context_in);

    // UKF Setup
    UKF ukf(Pointer<UKFMemory>(problem.GetMemory().pointer, problem.GetMemory().type, problem.GetMemory().context), alpha, beta, kappa);

    // Timer Setup
    Pointer<Timer> timer = MemoryHandler::AllocValue<Timer, unsigned>(UKF_TIMER + 1u, PointerType::CPU, PointerContext::CPU_Only);
    double *timer_pointer = timer.pointer[0u].GetValues();

    // Output Pointers
    Pointer<double> temperature_out = problem.GetMemory().pointer[0u].GetState().pointer[0u].GetPointer();
    Pointer<double> heatFlux_out = Pointer<double>(temperature_out.pointer + Lxyz, temperature_out.type, temperature_out.context);
    Pointer<double> temperatureMeasured_out = problem.GetMemory().pointer[0u].GetMeasure().pointer[0u].GetPointer();
    Pointer<double> covariance_out = problem.GetMemory().pointer[0u].GetStateCovariance().pointer[0u].GetPointer();

    Pointer<double> temperature_parser, heatFlux_parser, temperatureMeasured_parser, covariance_parser, error_parser;
    error_parser = MemoryHandler::Alloc<double>(L, PointerType::CPU, PointerContext::CPU_Only);
    Pointer<double> temperatureError_parser = Pointer<double>(error_parser.pointer, error_parser.type, error_parser.context);
    Pointer<double> heatFluxError_parser = Pointer<double>(error_parser.pointer + Lxyz, error_parser.type, error_parser.context);

    if (type_in == PointerType::GPU)
    {
        temperature_parser = MemoryHandler::Alloc<double>(Lxyz, PointerType::CPU, PointerContext::CPU_Only);
        heatFlux_parser = MemoryHandler::Alloc<double>(Lxy, PointerType::CPU, PointerContext::CPU_Only);
        temperatureMeasured_parser = MemoryHandler::Alloc<double>(Lxy, PointerType::CPU, PointerContext::CPU_Only);
        covariance_parser = MemoryHandler::Alloc<double>(L * L, PointerType::CPU, PointerContext::CPU_Only);
        MemoryHandler::Copy(temperature_parser, temperature_out, Lxyz);
        MemoryHandler::Copy(heatFlux_parser, heatFlux_out, Lxy);
        MemoryHandler::Copy(temperatureMeasured_parser, temperatureMeasured_out, Lxy);
        MemoryHandler::Copy(covariance_parser, covariance_out, L * L);
    }
    else
    {
        temperature_parser = temperature_out;
        heatFlux_parser = heatFlux_out;
        temperatureMeasured_parser = temperatureMeasured_out;
        covariance_parser = covariance_out;
    }
    Math::Diag(error_parser, covariance_parser, L, L, L, 0u, 0u);

    // Export initial values
    Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperature), Lxyz, ParserType::Double, temperature_parser.pointer, positionTemperature, 0u);
    Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexHeatFlux), Lxy, ParserType::Double, heatFlux_parser.pointer, positionHeatFlux, 0u);
    Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperatureMeasured), Lxy, ParserType::Double, temperatureMeasured_parser.pointer, positionTemperatureMeasured, 0u);
    Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperatureError), Lxyz, ParserType::Double, temperatureError_parser.pointer, positionTemperatureError, 0u);
    Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexHeatFluxError), Lxy, ParserType::Double, heatFlux_parser.pointer, positionHeatFluxError, 0u);
    Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexCovariance), L * L, ParserType::Double, covariance_parser.pointer, positionCovariance, 0u);

    for (unsigned i = 1u; i <= Lt; i++)
    {
        std::cout << "Iteration " << i << ":\n";
        problem.UpdateMeasure(generator.GetTemperature(i), Lx, Ly, type_in, context_in);
        ukf.Iterate(timer.pointer[0u]);

        // Export Values
        if (type_in == PointerType::GPU)
        {
            MemoryHandler::Copy(temperature_parser, temperature_out, Lxyz);
            MemoryHandler::Copy(heatFlux_parser, heatFlux_out, Lxy);
            MemoryHandler::Copy(temperatureMeasured_parser, temperatureMeasured_out, Lxy);
            MemoryHandler::Copy(covariance_parser, covariance_out, L * L);
            cudaDeviceSynchronize();
        }
        Math::Diag(error_parser, covariance_parser, L, L, L, 0u, 0u);

        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperature), Lxyz, ParserType::Double, temperature_parser.pointer, positionTemperature, i);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexHeatFlux), Lxy, ParserType::Double, heatFlux_parser.pointer, positionHeatFlux, i);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperatureError), Lxyz, ParserType::Double, temperatureError_parser.pointer, positionTemperatureError, i);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexHeatFluxError), Lxy, ParserType::Double, heatFlux_parser.pointer, positionHeatFluxError, i);
        timer.pointer[0u].Save();
        timer.pointer[0u].SetValues();
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTimer), UKF_TIMER + 1, ParserType::Double, timer_pointer, positionTimer, i - 1u);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperatureMeasured), Lxy, ParserType::Double, temperatureMeasured_parser.pointer, positionTemperatureMeasured, i);
    }
    Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexCovariance), L * L, ParserType::Double, covariance_parser.pointer, positionCovariance, 1u);

    MemoryHandler::Free<Parser>(parser);
    MemoryHandler::Free<Timer>(timer);
    MemoryHandler::Free(error_parser);
    if (type_in == PointerType::GPU)
    {
        MemoryHandler::Free(temperature_parser);
        MemoryHandler::Free(heatFlux_parser);
        MemoryHandler::Free(temperatureMeasured_parser);
        MemoryHandler::Free(covariance_parser);
    }
    return 0;
}

int main(int argc, char **argv)
{
    std::cout << "\nStart Execution\n\n";

    unsigned Lx;
    unsigned Ly;
    unsigned Lz;
    unsigned Lt;

    PointerType pointerType;
    PointerContext pointerContext;

    if (argc == 1 + 4 + 2)
    {
        Lx = (unsigned)std::stoi(argv[1u]);
        Ly = (unsigned)std::stoi(argv[2u]);
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

    std::string name_size = "ParameterSize";
    std::string name_heatProblem = "ParameterHeatProblem";
    std::string name_UKFParm = "ParameterUKF";

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

    double Sx = S[0u];
    double Sy = S[1u];
    double Sz = S[2u];
    double St = S[3u];

    double T0 = HPParm[0u];
    double sT0 = HPParm[1u];
    double sTm0 = HPParm[2u];
    double Q0 = HPParm[3u];
    double sQ0 = HPParm[4u];
    double Amp = HPParm[5u];
    double mean = HPParm[6u];
    double sigma = HPParm[7u];

    double alpha = UKFParm[0u];
    double beta = UKFParm[1u];
    double kappa = UKFParm[2u];

    delete[] S;
    delete[] HPParm;
    delete[] UKFParm;

    std::cout << "\nRunning case with grid: (" << Lx << "," << Ly << "," << Lz << "," << Lt << ")\n\n";

    RunCase(path_binary_out, extension_binary,
            Lx, Ly, Lz, Lt,
            Sx, Sy, Sz, St,
            T0, sT0, sTm0, Q0, sQ0, Amp, mean, sigma,
            alpha, beta, kappa,
            pointerType, pointerContext);

    Parser::ConvertToText(path_binary_out, path_text_out, extension_text);

    std::string name_type = (pointerType == PointerType::GPU) ? "_GPU" : "_CPU";
    std::string ok_name = path_text_out +
                          "X" + std::to_string(Lx) +
                          "Y" + std::to_string(Ly) +
                          "Z" + std::to_string(Lz) +
                          "T" + std::to_string(Lt) +
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