#include <iostream>
#include "structure/include/Data.hpp"
#include "structure/include/DataCovariance.hpp"
#include "structure/include/Parameter.hpp"
#include "parser/include/Parser.hpp"
#include "ukf/include/UKF.hpp"
#include "timer/include/Timer.hpp"
#include "hfe/include/HeatFluxEstimation.hpp"
#include "hfe/include/HeatFluxGenerator.hpp"
#include <stdlib.h>
#include <filesystem>

int RunCase(std::string& path_binary, std::string& extension_binary,
    unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt,
    double Sx, double Sy, double Sz, double St,
    double T0, double Amp, double mean, double sigma,
    double alpha, double beta, double kappa)
{
    Pointer<Parser> parser = MemoryHandler::AllocValue<Parser,unsigned>(4u,PointerType::CPU,PointerContext::CPU_Only);

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

    std::string name_timer_aux = name_timer 
    + "X" + std::to_string(Lx) 
    + "Y" + std::to_string(Ly) 
    + "Z" + std::to_string(Lz) 
    + "T" + std::to_string(Lt);
    std::string name_temperature_aux = name_temperature
    + "X" + std::to_string(Lx) 
    + "Y" + std::to_string(Ly) 
    + "Z" + std::to_string(Lz) 
    + "T" + std::to_string(Lt);
    std::string name_temperature_measured_aux = name_temperature_measured
    + "X" + std::to_string(Lx) 
    + "Y" + std::to_string(Ly) 
    + "Z" + std::to_string(Lz) 
    + "T" + std::to_string(Lt);
    std::string name_heatFlux_aux = name_heatFlux
    + "X" + std::to_string(Lx) 
    + "Y" + std::to_string(Ly) 
    + "Z" + std::to_string(Lz) 
    + "T" + std::to_string(Lt);

    indexTimer = parser.pointer[0u].OpenFileOut(path_binary,name_timer_aux,extension_binary,std::ios::binary);
    indexTemperature = parser.pointer[0u].OpenFileOut(path_binary,name_temperature_aux,extension_binary,std::ios::binary);
    indexTemperatureMeasured = parser.pointer[0u].OpenFileOut(path_binary,name_temperature_measured_aux,extension_binary,std::ios::binary);
    indexHeatFlux = parser.pointer[0u].OpenFileOut(path_binary,name_heatFlux_aux,extension_binary,std::ios::binary);

    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTimer),"Timer",UKF_TIMER+1,ParserType::Double,positionTimer);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTemperature),"Temperature",Lx*Ly*Lz,ParserType::Double,positionTemperature);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexTemperatureMeasured),"Temperature Measured",Lx*Ly,ParserType::Double,positionTemperatureMeasured);
    Parser::ExportConfigurationBinary(parser.pointer[0u].GetStreamOut(indexHeatFlux),"Heat Flux",Lx*Ly,ParserType::Double,positionHeatFlux);

    HeatFluxGenerator generator(Lx,Ly,Lz,Lt,Sx,Sy,Sz,St,T0,Amp);
    generator.Generate(mean,sigma);

    HeatFluxEstimation problem(Lx,Ly,Lz,Lt,Sx,Sy,Sz,St);

    problem.UpdateMeasure(generator.GetTemperature(0u),Lx,Ly);
    Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperatureMeasured),Lx*Ly,ParserType::Double,generator.GetTemperature(0u).pointer,positionTemperatureMeasured,0u);
        
    UKF ukf(Pointer<UKFMemory>(problem.GetMemory().pointer, problem.GetMemory().type, problem.GetMemory().context), alpha, beta, kappa);

    //Math::PrintMatrix(generator.GetTemperature(Lt),Lx,Ly);

    Pointer<Timer> timer = MemoryHandler::AllocValue<Timer,unsigned>(UKF_TIMER+1u,PointerType::CPU, PointerContext::CPU_Only);
    for(unsigned i = 1u; i <= Lt; i++){
        std::cout << "Iteration " << i << ":\n";
        ukf.Iterate(timer.pointer[0u]);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperature),Lx*Ly*Lz,ParserType::Double,problem.GetMemory().pointer[0u].GetState().pointer[0u].GetPointer().pointer,positionTemperature,i-1u);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexHeatFlux),Lx*Ly,ParserType::Double,problem.GetMemory().pointer[0u].GetState().pointer[0u].GetPointer().pointer+Lx*Ly*Lz,positionHeatFlux,i-1u);
        timer.pointer[0u].Save();
        timer.pointer[0u].SetValues();
        //timer->Print();
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTimer),UKF_TIMER+1,ParserType::Double,timer.pointer[0u].GetValues(),positionTimer,i-1u);
        //Math::PrintMatrix(problem.GetMemory()->GetState()->GetPointer(),Lx,Ly);
        //Math::PrintMatrix(problem.GetMemory()->GetState()->GetPointer()+Lx*Ly*Lz,Lx,Ly);
        problem.UpdateMeasure(generator.GetTemperature(i),Lx,Ly);
        Parser::ExportValuesBinary(parser.pointer[0u].GetStreamOut(indexTemperatureMeasured),Lx*Ly,ParserType::Double,generator.GetTemperature(i).pointer,positionTemperatureMeasured,i);
    }

    MemoryHandler::Free<Parser>(parser);
    MemoryHandler::Free<Timer>(timer);

    return 0;
}

int main(int argc, char** argv){
    std::cout << "\nStart Execution\n\n";
    
    unsigned Lx; 
    unsigned Ly;
    unsigned Lz;
    unsigned Lt;

    if(argc == 1 + 4) {
        Lx = (unsigned)std::stoi(argv[1u]);
        Ly = (unsigned)std::stoi(argv[2u]);
        Lz = (unsigned)std::stoi(argv[3u]);
        Lt = (unsigned)std::stoi(argv[4u]);
    } else {
        std::cout << "Error: Wrong number of parameters: " << argc << "\n";
        return 1;
    }

    std::string path_dir = std::filesystem::current_path();
    std::string path_text_in = path_dir + "/data/text/in/";
    std::string path_binary_in = path_dir + "/data/binary/in/";
    
    std::string path_text_out = path_dir + "/data/text/out/";
    std::string path_binary_out = path_dir + "/data/binary/out/";

    std::string name_size = "ParameterSize";
    std::string name_heatProblem = "ParameterHeatProblem";
    std::string name_UKFParm = "ParameterUKF";

    std::string extension_text = ".dat";
    std::string extension_binary = ".bin";

    Parser::ConvertToBinary(path_text_in,path_binary_in,extension_binary);

    Parser* parser = new Parser(20u);

    unsigned indexSize = parser->OpenFileIn(path_binary_in,name_size,extension_binary,std::ios::binary);
    unsigned indexHPParm = parser->OpenFileIn(path_binary_in,name_heatProblem,extension_binary,std::ios::binary);
    unsigned indexUKFParm = parser->OpenFileIn(path_binary_in,name_UKFParm,extension_binary,std::ios::binary);

    double* S = NULL;
    double* HPParm = NULL;
    double* UKFParm = NULL;

    std::string name;
    unsigned length;
    ParserType type;
    unsigned iteration;
    void* pointer = NULL;
    
    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexSize),name,length,type,iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexSize),length,type,pointer,0u);
    S = (double*)pointer;
    
    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexHPParm),name,length,type,iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexHPParm),length,type,pointer,0u);
    HPParm = (double*)pointer;
    
    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexUKFParm),name,length,type,iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexUKFParm),length,type,pointer,0u);
    UKFParm = (double*)pointer;

    delete parser;

    double Sx = S[0u];
    double Sy = S[1u];
    double Sz = S[2u];
    double St = S[3u];

    double T0 = HPParm[0u];
    double Amp = HPParm[1u];
    double mean = HPParm[2u];
    double sigma = HPParm[3u];
    
    double alpha = UKFParm[0u];
    double beta = UKFParm[1u];
    double kappa = UKFParm[2u];

    delete[] S;
    delete[] HPParm;
    delete[] UKFParm;

    std::cout << "\nRunning case with grid: (" << Lx << "," << Ly << "," << Lz << "," << Lt << ")\n\n";

    RunCase(path_binary_out,extension_binary,
        Lx,Ly,Lz,Lt,
        Sx,Sy,Sz,St,
        T0,Amp,mean,sigma,
        alpha,beta,kappa
    );

    Parser::ConvertToText(path_binary_out,path_text_out,extension_text);

    std::string ok_name = path_text_out +
        "X" + std::to_string(Lx) +
        "Y" + std::to_string(Ly) +
        "Z" + std::to_string(Lz) +
        "T" + std::to_string(Lt) + ".ok";
    std::ofstream ok_file(ok_name);
    ok_file.close();

    std::cout << "\nEnd Execution\n";
    return 0;
}