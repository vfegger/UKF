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
    Parser* parser = new Parser(4u);

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

    indexTimer = parser->OpenFileOut(path_binary,name_timer_aux,extension_binary,std::ios::binary);
    indexTemperature = parser->OpenFileOut(path_binary,name_temperature_aux,extension_binary,std::ios::binary);
    indexTemperatureMeasured = parser->OpenFileOut(path_binary,name_temperature_measured_aux,extension_binary,std::ios::binary);
    indexHeatFlux = parser->OpenFileOut(path_binary,name_heatFlux_aux,extension_binary,std::ios::binary);

    Parser::ExportConfigurationBinary(parser->GetStreamOut(indexTimer),"Timer",UKF_TIMER+1,ParserType::Double,positionTimer);
    Parser::ExportConfigurationBinary(parser->GetStreamOut(indexTemperature),"Temperature",Lx*Ly*Lz,ParserType::Double,positionTemperature);
    Parser::ExportConfigurationBinary(parser->GetStreamOut(indexTemperatureMeasured),"Temperature Measured",Lx*Ly,ParserType::Double,positionTemperatureMeasured);
    Parser::ExportConfigurationBinary(parser->GetStreamOut(indexHeatFlux),"Heat Flux",Lx*Ly,ParserType::Double,positionHeatFlux);

    HeatFluxGenerator generator(Lx,Ly,Lz,Lt,Sx,Sy,Sz,St,T0,Amp);
    generator.Generate(mean,sigma);

    HeatFluxEstimation problem(Lx,Ly,Lz,Lt,Sx,Sy,Sz,St);

    problem.UpdateMeasure(generator.GetTemperature(0u),Lx,Ly);
    Parser::ExportValuesBinary(parser->GetStreamOut(indexTemperatureMeasured),Lx*Ly,ParserType::Double,generator.GetTemperature(0u),positionTemperatureMeasured,0u);
        
    UKF ukf(problem.GetMemory(), alpha, beta, kappa);

    Math::PrintMatrix(generator.GetTemperature(Lt),Lx,Ly);

    Timer* timer = new Timer(UKF_TIMER+1u);
    for(unsigned i = 1u; i <= Lt; i++){
        std::cout << "Iteration " << i << ":\n";
        ukf.Iterate(*timer);
        Parser::ExportValuesBinary(parser->GetStreamOut(indexTemperature),Lx*Ly*Lz,ParserType::Double,problem.GetMemory()->GetState()->GetPointer(),positionTemperature,i-1u);
        Parser::ExportValuesBinary(parser->GetStreamOut(indexHeatFlux),Lx*Ly,ParserType::Double,problem.GetMemory()->GetState()->GetPointer()+Lx*Ly*Lz,positionHeatFlux,i-1u);
        timer->Save();
        timer->SetValues();
        timer->Print();
        Parser::ExportValuesBinary(parser->GetStreamOut(indexTimer),UKF_TIMER+1,ParserType::Double,timer->GetValues(),positionTimer,i-1u);
        Math::PrintMatrix(problem.GetMemory()->GetState()->GetPointer(),Lx,Ly);
        Math::PrintMatrix(problem.GetMemory()->GetState()->GetPointer()+Lx*Ly*Lz,Lx,Ly);
        problem.UpdateMeasure(generator.GetTemperature(i),Lx,Ly);
        Parser::ExportValuesBinary(parser->GetStreamOut(indexTemperatureMeasured),Lx*Ly,ParserType::Double,generator.GetTemperature(i),positionTemperatureMeasured,i);
    }

    delete parser;
    delete timer;

    return 0;
}

int main(int argc, char** argv){
    std::cout << "\nStart Execution\n\n";
    std::string path_dir = std::filesystem::current_path();
    std::string path_text_in = path_dir + "/data/text/in/";
    std::string path_binary_in = path_dir + "/data/binary/in/";
    
    std::string path_text_out = path_dir + "/data/text/out/";
    std::string path_binary_out = path_dir + "/data/binary/out/";

    std::string name_length = "ParameterLength";
    std::string name_size = "ParameterSize";
    std::string name_heatProblem = "ParameterHeatProblem";
    std::string name_UKFParm = "ParameterUKF";

    std::string extension_text = ".dat";
    std::string extension_binary = ".bin";

    Parser::ConvertToBinary(path_text_in,path_binary_in,extension_binary);

    Parser* parser = new Parser(20u);

    unsigned indexLength = parser->OpenFileIn(path_binary_in,name_length,extension_binary,std::ios::binary);
    unsigned indexSize = parser->OpenFileIn(path_binary_in,name_size,extension_binary,std::ios::binary);
    unsigned indexHPParm = parser->OpenFileIn(path_binary_in,name_heatProblem,extension_binary,std::ios::binary);
    unsigned indexUKFParm = parser->OpenFileIn(path_binary_in,name_UKFParm,extension_binary,std::ios::binary);

    unsigned* L_lower = NULL, * L_upper = NULL, * L_ref = NULL, * L_stride = NULL;
    double* S = NULL;
    double* HPParm = NULL;
    double* UKFParm = NULL;

    std::string name;
    unsigned length;
    ParserType type;
    unsigned iteration;
    void* pointer = NULL;

    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexLength),name,length,type,iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexLength),length,type,pointer,0u);
    L_ref = (unsigned*)pointer;
    Parser::ImportValuesBinary(parser->GetStreamIn(indexLength),length,type,pointer,1u);
    L_lower = (unsigned*)pointer;
    Parser::ImportValuesBinary(parser->GetStreamIn(indexLength),length,type,pointer,2u);
    L_upper = (unsigned*)pointer;
    Parser::ImportValuesBinary(parser->GetStreamIn(indexLength),length,type,pointer,3u);
    L_stride = (unsigned*)pointer;
    
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

    unsigned Lx_ref = L_ref[0u];
    unsigned Ly_ref = L_ref[1u];
    unsigned Lz_ref = L_ref[2u];
    unsigned Lt_ref = L_ref[3u];
    unsigned Lx_lower = L_lower[0u];
    unsigned Ly_lower = L_lower[1u];
    unsigned Lz_lower = L_lower[2u];
    unsigned Lt_lower = L_lower[3u];
    unsigned Lx_upper = L_upper[0u];
    unsigned Ly_upper = L_upper[1u];
    unsigned Lz_upper = L_upper[2u];
    unsigned Lt_upper = L_upper[3u];
    unsigned Lx_stride = L_stride[0u];
    unsigned Ly_stride = L_stride[1u];
    unsigned Lz_stride = L_stride[2u];
    unsigned Lt_stride = L_stride[3u];

    unsigned Lx = Lx_ref; 
    unsigned Ly = Ly_ref;
    unsigned Lz = Lz_ref;
    unsigned Lt = Lt_ref;
    if(argc == 1 + 4) {
        int cx = std::stoi(argv[1u]);
        int cy = std::stoi(argv[2u]);
        int cz = std::stoi(argv[3u]);
        int ct = std::stoi(argv[4u]);
        Lx = (Lx_upper > Lx_lower + Lx_stride * cx) ? Lx_lower + Lx_stride * cx : Lx_upper; 
        Ly = (Ly_upper > Ly_lower + Ly_stride * cy) ? Ly_lower + Ly_stride * cy : Ly_upper;
        Lz = (Lz_upper > Lz_lower + Lz_stride * cz) ? Lz_lower + Lz_stride * cz : Lz_upper;
        Lt = (Lt_upper > Lt_lower + Lt_stride * ct) ? Lt_lower + Lt_stride * ct : Lt_upper;
    }

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

    delete[] L_lower;
    delete[] L_upper;
    delete[] L_ref;
    delete[] L_stride;
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

    std::cout << "\nEnd Execution\n";
    return 0;
}