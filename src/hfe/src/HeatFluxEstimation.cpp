#include "../include/HeatFluxEstimation.hpp"


HeatFluxEstimationMemory* HeatFluxEstimation::GetMemory(){
    return memory;
}

HeatFluxEstimation::HeatFluxEstimation(
    unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt,
    double Sx, double Sy, double Sz, double St){
        std::cout << "Parameter Initialization\n";
        parameter = new Parameter(4u);
        unsigned indexL, indexD, indexS, indexAmp;
        indexL = parameter->Add("Length",4u,sizeof(unsigned));
        indexD = parameter->Add("Delta",4u,sizeof(double));
        indexS = parameter->Add("Size",4u,sizeof(double));
        indexAmp = parameter->Add("Amp",1u,sizeof(double));
        parameter->Initialize();
        unsigned L[4] = {Lx,Ly,Lz,Lt};
        parameter->LoadData(indexL, L, 4u);
        double D[4] = {Sx/Lx,Sy/Ly,Sz/Lz,St/Lt};
        parameter->LoadData(indexD, D, 4u);
        double S[4] = {Sx,Sy,Sz,St};
        parameter->LoadData(indexS, S, 4u);
        double Amp[1] = {5e4};
        parameter->LoadData(indexAmp, Amp, 1u);
        std::cout << "Input Initialization\n";
        input = new Data(2u);
        unsigned indexT, indexQ;
        indexT = input->Add("Temperature",Lx*Ly*Lz);
        indexQ = input->Add("Heat Flux",Lx*Ly);
        input->Initialize();
        double T[Lx*Ly*Lz];
        double sigmaT[Lx*Ly*Lz];
        for(unsigned i = 0u; i < Lx*Ly*Lz; i++){
            T[i] = 300.0;
            sigmaT[i] = 1.0;
        }
        double Q[Lx*Ly];
        double sigmaQ[Lx*Ly];
        for(unsigned i = 0u; i < Lx*Ly; i++){
            Q[i] = 0.0;
            sigmaQ[i] = 1.25;
        }
        input->LoadData(indexT, T, Lx*Ly*Lz);
        input->LoadData(indexQ, Q, Lx*Ly);
        inputCovariance = new DataCovariance(*input);
        inputNoise = new DataCovariance(*input);
        inputCovariance->LoadData(indexT, sigmaT, Lx*Ly*Lz, DataCovarianceMode::Compact);
        inputCovariance->LoadData(indexQ, sigmaQ, Lx*Ly, DataCovarianceMode::Compact);
        inputNoise->LoadData(indexT, sigmaT, Lx*Ly*Lz, DataCovarianceMode::Compact);
        inputNoise->LoadData(indexQ, sigmaQ, Lx*Ly, DataCovarianceMode::Compact);
        
        std::cout << "Measure Initialization\n";
        measure = new Data(1u);
        unsigned indexT_meas;
        indexT_meas = measure->Add("Temperature",Lx*Ly);
        measure->Initialize();
        double T_meas[Lx*Ly];
        double sigmaT_meas[Lx*Ly];
        for(unsigned i = 0u; i < Lx*Ly; i++){
            T_meas[i] = 300.0;
            sigmaT_meas[i] = 1.5;
        }
        measure->LoadData(indexT_meas, T_meas, Lx*Ly);
        measureNoise = new DataCovariance(*measure);
        measureNoise->LoadData(indexT_meas, sigmaT_meas, Lx*Ly, DataCovarianceMode::Compact);

        std::cout << "Memory Initialization\n";
        memory = new HeatFluxEstimationMemory(*input,*inputCovariance,*inputNoise,*measure,*measureNoise,*parameter);
        
        std::cout << "End Initialization\n";
}

void HeatFluxEstimation::UpdateMeasure(double* T_in, unsigned Lx, unsigned Ly){
    Data* measure_aux = new Data(1u);
    unsigned indexT_meas;
    indexT_meas = measure_aux->Add("Temperature",Lx*Ly);
    measure_aux->Initialize();
    measure_aux->LoadData(indexT_meas, T_in, Lx*Ly); 
    memory->UpdateMeasure(*measure_aux);
    delete measure_aux;
}

HeatFluxEstimation::~HeatFluxEstimation(){
    delete memory;
    delete parameter;
    delete input;
    delete inputCovariance;
    delete inputNoise;
    delete measure;
    delete measureNoise;
}
