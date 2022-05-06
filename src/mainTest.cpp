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
    std::string path_text = "/mnt/d/Research/UKF/data/text/in/";
    std::string path_binary = "/mnt/d/Research/UKF/data/binary/in/";

    Parser::ConvertToBinary(path_text,path_binary);

    return 1;

    unsigned Lx = 12u;
    unsigned Ly = 12u;
    unsigned Lz = 6u;
    unsigned Lt = 100u;
    double Sx = 0.12;
    double Sy = 0.12;
    double Sz = 0.003;
    double St = 2.0;
    double T0 = 300.0;
    double Amp = 5.0e4;
    double mean = 0.0;
    double sigma = 1.5;

    HeatFluxGenerator generator(Lx,Ly,Lz,Lt,Sx,Sy,Sz,St,T0,Amp);
    
    generator.Generate(mean,sigma);

    HeatFluxEstimation problem(Lx,Ly,Lz,Lt,Sx,Sy,Sz,St);
    
    problem.UpdateMeasure(generator.GetTemperature(0u),Lx,Ly);
    
    double alpha = 0.001;
    double beta = 2.0;
    double kappa = 0.0;
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