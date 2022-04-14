#include <iostream>
#include "structure/include/Data.hpp"
#include "structure/include/DataCovariance.hpp"
#include "structure/include/Parameter.hpp"
#include "parser/include/Parser.hpp"
#include "ukf/include/UKF.hpp"

class HeatFluxEstimationMemory : public UKFMemory{
public:
    HeatFluxEstimationMemory(Data& a, DataCovariance&b, DataCovariance&c, Data&d, DataCovariance&e, Parameter&f) : UKFMemory(a,b,c,d,e,f){

    }

    inline double C(double T){
        return 1324.75*T+3557900.0;
    }
    inline double K(double T){
        return 12.45 + (14e-3 + 2.517e-6*T)*T;
    }

    unsigned DifferentialK(double TN, double T, double TP, double delta){
        double auxN = 2.0*(K(T)*K(TN))/(K(T)+K(TN))*(TN - T)/delta;
        double auxP = 2.0*(K(TP)*K(T))/(K(TP)+K(T))*(TP - T)/delta;
        return (auxN+auxP)/delta;
    }

    void Evolution(Data& data_inout, Parameter& parameter_in) override {
        unsigned* length = parameter_in.GetPointer<unsigned>(0u);
        unsigned Lx = length[0u];
        unsigned Ly = length[1u];
        unsigned Lz = length[2u];
        unsigned Lt = length[3u];
        double* delta = parameter_in.GetPointer<double>(1u);
        double dx = delta[0u];
        double dy = delta[1u];
        double dz = delta[2u];
        double dt = delta[3u];
        unsigned acc = 0.0;
        double TiN, TiP;
        double TjN, TjP;
        double TkN, TkP;
        double T0;
        unsigned index;
        double* T = data_inout[0u];
        double* Q = data_inout[1u];
        for(unsigned k = 0u; k < Lz; k++){
            for(unsigned j = 0u; j < Ly; j++){
                for(unsigned i = 0u; i < Lx; i++){
                    index = (k*Ly+j)*Lx+i;
                    T0 = T[index];
                    TiN = (i != 0u  ) ? T[index - 1]  : T0;
                    TiP = (i != Lx-1) ? T[index + 1]  : T0;
                    TjN = (j != 0u  ) ? T[index - Lx] : T0;
                    TjP = (j != Ly-1) ? T[index + Lx] : T0;
                    TkN = (k != 0u  ) ? T[index - Ly] : T0;
                    TkP = (k != Lz-1) ? T[index + Ly] : T0;
                    acc = 0.0;
                    // X dependency
                    acc += DifferentialK(TiN,T0,TiP,dx);
                    // Y dependency
                    acc += DifferentialK(TjN,T0,TjP,dx);
                    // Z dependency
                    acc += DifferentialK(TkN,T0,TkP,dx);

                    if(k == Lz - 1){
                        acc += Q[j*Lx+i]/dz;
                    }
                    T[index] += dt*acc/C(T0);
                }
            }
        }

    }
    void Observation(Data& data_in, Parameter& parameter_in, Data& data_out) override {
        unsigned* length = parameter_in.GetPointer<unsigned>(0u);
        unsigned Lx = length[0u];
        unsigned Ly = length[1u];
        double* T_in = data_in[0u];
        double* T_out = data_out[0u];
        unsigned index;
        for(unsigned j = 0u; j < Ly; j++){
            for(unsigned i = 0u; i < Lx; i++){
                index = j*Lx+i;
                T_out[index] = T_in[index];
            }
        }
    }
};

class HeatFluxEstimation {
private:
    HeatFluxEstimationMemory* memory;
    Parameter* parameter;
    Data* input;
    DataCovariance* inputCovariance;
    DataCovariance* inputNoise;
    Data* measure;
    DataCovariance* measureNoise;
public:
    HeatFluxEstimationMemory* GetMemory(){
        return memory;
    }

    HeatFluxEstimation(
        unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt,
        double Sx, double Sy, double Sz, double St){
            std::cout << "Parameter Initialization\n";
            parameter = new Parameter(3u);
            unsigned indexL, indexD, indexS;
            indexL = parameter->Add("Length",4u,sizeof(unsigned));
            indexD = parameter->Add("Delta",4u,sizeof(double));
            indexS = parameter->Add("Size",4u,sizeof(double));
            parameter->Initialize();
            unsigned L[4] = {Lx,Ly,Lz,Lt};
            parameter->LoadData(indexL, L, 4u);
            double D[4] = {Sx/Lx,Sy/Ly,Sz/Lz,St/Lt};
            parameter->LoadData(indexD, D, 4u);
            double S[4] = {Sx,Sy,Sz,St};
            parameter->LoadData(indexS, S, 4u);

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
                Q[i] = 10.0;
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

    ~HeatFluxEstimation(){
        delete memory;
        delete parameter;
        delete input;
        delete inputCovariance;
        delete inputNoise;
        delete measure;
        delete measureNoise;
    }

};

int main(){
    std::cout << "\nStart Execution\n\n";
    std::string path = "/mnt/d/Research/UKF/data/";

    Data a(10u);
    unsigned i = a.Add("Test_Data",3);
    a.Initialize();
    double b[3] = {1.1,2.2,3.3};
    a.LoadData(i, b, 3);

    std::cout << "Pointer: " << a.GetPointer() << "\n";
    for(unsigned j = 0u; j < 3; j++){
        std::cout << "Values: " << a[i][j] << "\n";
    }

    Parameter c(10u);
    unsigned k = c.Add("Test_Parm",3,sizeof(unsigned));
    c.Initialize();
    unsigned d[3] = {4u,5u,6u};
    c.LoadData(k,d,3);
    std::cout << "Pointer: " << c.GetPointer<unsigned>(k) << "\n";
    for(unsigned j = 0u; j < 3; j++){
        std::cout << "Values: " << c.GetPointer<unsigned>(k)[j] << "\n";
    }

    Parser parser(3u);
    unsigned index = parser.OpenFile(path + "TestData",".dat");
    std::string name = "";
    unsigned length = 0u;
    parser.ImportConfiguration(index,name,length);
    std::cout << "Name: " << name << "\n";
    std::cout << "Length: " << length << "\n";
    double values[length];
    parser.ImportData(index, length, values);
    for(unsigned j = 0u; j < length; j++){
        std::cout << "Values: " << values[j] << "\n";
    }

    std::cout << "Test: Data Covariance \n"; 
    DataCovariance e = DataCovariance(a);
    std::cout << "Pointer: " << e.GetPointer() << "\n";
    for(unsigned j = 0u; j < 3; j++){
        e[i][j*3+j] = 1.0;
    } 
    for(unsigned j = 0u; j < 9; j++){
        std::cout << "Values: " << e[i][j] << "\n";
    } 

    HeatFluxEstimation problem(4,4,2,20,160.0,160.0,3.0,2.0);

    UKF ukf(problem.GetMemory(), 0.001, 3.0, 0.0);

    Math::PrintMatrix(problem.GetMemory()->GetState()->GetPointer(),4*4*(2+1),1);

    ukf.Iterate();

    Math::PrintMatrix(problem.GetMemory()->GetState()->GetPointer(),4*4*(2+1),1);

    std::cout << "\nEnd Execution\n";
    return 0;
}