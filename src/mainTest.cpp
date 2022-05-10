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

int main(){
    std::cout << "\nStart Execution\n\n";
    std::string path_text_in = "/mnt/d/Research/UKF/data/text/in/";
    std::string path_binary_in = "/mnt/d/Research/UKF/data/binary/in/";
    
    std::string path_text_out = "/mnt/d/Research/UKF/data/text/out/";
    std::string path_binary_out = "/mnt/d/Research/UKF/data/binary/out/";

    std::string name_length = "ParameterLength";
    std::string name_size = "ParameterSize";
    std::string name_heatProblem = "ParameterHeatProblem";
    std::string name_UKFParm = "ParameterUKF";

    std::string name_timer = "Timer";
    std::string name_temperature = "Temperature";
    std::string name_heatFlux = "HeatFlux";

    std::string extension_text = ".dat";
    std::string extension_binary = ".bin";

    Parser::ConvertToBinary(path_text_in,path_binary_in,extension_binary);
    Parser::ConvertToText(path_binary_out,path_text_out,extension_text);

    Parser* parser = new Parser(20u);

    unsigned indexLength = parser->OpenFileIn(path_binary_in,name_length,extension_binary,std::ios::binary);
    unsigned indexSize = parser->OpenFileIn(path_binary_in,name_size,extension_binary,std::ios::binary);
    unsigned indexHPParm = parser->OpenFileIn(path_binary_in,name_heatProblem,extension_binary,std::ios::binary);
    unsigned indexUKFParm = parser->OpenFileIn(path_binary_in,name_UKFParm,extension_binary,std::ios::binary);

    unsigned* L_lower = NULL, * L_upper = NULL;
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
    L_lower = (unsigned*)pointer;
    Parser::ImportValuesBinary(parser->GetStreamIn(indexLength),length,type,pointer,1u);
    L_upper = (unsigned*)pointer;
    
    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexSize),name,length,type,iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexSize),length,type,pointer,0u);
    S = (double*)pointer;
    
    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexHPParm),name,length,type,iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexHPParm),length,type,pointer,0u);
    HPParm = (double*)pointer;
    
    Parser::ImportConfigurationBinary(parser->GetStreamIn(indexHPParm),name,length,type,iteration);
    Parser::ImportValuesBinary(parser->GetStreamIn(indexHPParm),length,type,pointer,0u);
    UKFParm = (double*)pointer;

    unsigned Lx = L_lower[0u];
    unsigned Ly = L_lower[1u];
    unsigned Lz = L_lower[2u];
    unsigned Lt = L_lower[3u];

    double Sx = S[0u];
    double Sy = S[1u];
    double Sz = S[2u];
    double St = S[3u];

    double T0 = HPParm[0u];
    double Amp = HPParm[1u];
    double mean = HPParm[2u];
    double sigma = HPParm[3u];

    HeatFluxGenerator generator(Lx,Ly,Lz,Lt,Sx,Sy,Sz,St,T0,Amp);
    
    generator.Generate(mean,sigma);

    HeatFluxEstimation problem(Lx,Ly,Lz,Lt,Sx,Sy,Sz,St);
    
    problem.UpdateMeasure(generator.GetTemperature(0u),Lx,Ly);
    
    double alpha = UKFParm[0u];
    double beta = UKFParm[1u];
    double kappa = UKFParm[2u];
    UKF ukf(problem.GetMemory(), alpha, beta, kappa);

    Math::PrintMatrix(generator.GetTemperature(Lt),Lx,Ly);

    Timer timer(UKF_TIMER);
    for(unsigned i = 1u; i <= Lt; i++){
        ukf.Iterate(timer);
        std::cout << "\n";
        Math::PrintMatrix(problem.GetMemory()->GetState()->GetPointer()+Lx*Ly*Lz,Lx,Ly);
        timer.Print();
        problem.UpdateMeasure(generator.GetTemperature(i),Lx,Ly);
    }

    std::cout << "\nEnd Execution\n";
    return 0;
}