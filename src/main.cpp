#include <iostream>
#include "Input.hpp"
#include "Output.hpp"
#include "UKF.hpp"
#include <stdlib.h>

class GravitationInput : public Input
{
private:
    double* position;
    double* velocity;
    double* acceleration;

    double* positionCovar;
    double* velocityCovar;
    double* accelerationCovar;

    double* velocityMeas;
    double* accelerationMeas;

    double deltaTime;
public:
    GravitationInput(double px, double py, double pz, double vx, double vy, double vz, double ax, double ay, double az, double dt){
        //State
        position = new double[3u];
        position[0u] = px;
        position[1u] = py;
        position[2u] = pz;

        velocity = new double[3u];
        velocity[0u] = 0.0;
        velocity[1u] = 0.0;
        velocity[2u] = 0.0;

        acceleration = new double[3u];
        acceleration[0u] = 0.0;
        acceleration[1u] = 0.0;
        acceleration[2u] = 0.0;

        //Covariance
        positionCovar = new double[3u];
        positionCovar[0u] = 0.1;
        positionCovar[1u] = 0.1;
        positionCovar[2u] = 0.1;

        velocityCovar = new double[3u];
        velocityCovar[0u] = 0.01;
        velocityCovar[1u] = 0.01;
        velocityCovar[2u] = 0.01;

        accelerationCovar = new double[3u];
        accelerationCovar[0u] = 0.01;
        accelerationCovar[1u] = 0.01;
        accelerationCovar[2u] = 0.01;

        //Measure
        velocityMeas = new double[3u];
        velocityMeas[0u] = vx + (double)(rand()%1000)/1000.0;
        velocityMeas[1u] = vy + (double)(rand()%1000)/1000.0;
        velocityMeas[2u] = vz + (double)(rand()%1000)/1000.0;

        accelerationMeas = new double[3u];
        accelerationMeas[0u] = ax + (double)(rand()%1000)/1000.0;
        accelerationMeas[1u] = ay + (double)(rand()%1000)/1000.0;
        accelerationMeas[2u] = az + (double)(rand()%1000)/1000.0;

        deltaTime = 1;
    }

    ~GravitationInput(){
        delete[] accelerationCovar;
        delete[] velocityCovar;
        delete[] positionCovar;
        delete[] acceleration;
        delete[] velocity;
        delete[] position;
        delete[] velocityMeas;
        delete[] accelerationMeas;
    }

    void LoadInput(){
        Data* inputData = new Data[3u];
        inputData[0u] = Data("Position", position, 3u);
        inputData[1u] = Data("Velocity", velocity, 3u);
        inputData[2u] = Data("Acceleration", acceleration, 3u);
        
        Data* inputDataCovar = new Data[3u];
        inputDataCovar[0u] = Data("PositionCovar", positionCovar, 3u);
        inputDataCovar[1u] = Data("VelocityCovar", velocityCovar, 3u);
        inputDataCovar[2u] = Data("AccelerationCovar", accelerationCovar, 3u);
        
        Data* inputDataNoise = new Data[3u];
        inputDataNoise[0u] = Data("PositionNoise", positionCovar, 3u);
        inputDataNoise[1u] = Data("VelocityNoise", velocityCovar, 3u);
        inputDataNoise[2u] = Data("AccelerationNoise", accelerationCovar, 3u);

        Data* measureData = new Data[2u];
        measureData[0u] = Data("VelocityMeasured", velocityMeas, 3u);
        measureData[1u] = Data("AccelerationMeasured", accelerationMeas, 3u);

        Data* measureDataNoise = new Data[2u];
        measureDataNoise[0u] = Data("VelocityMeasuredNoise", velocityMeas, 3u);
        measureDataNoise[1u] = Data("AccelerationMeasuredNoise", accelerationMeas, 3u);


        Initialize(inputData, inputDataCovar, inputDataNoise, 3u, NULL, 0, measureData, measureDataNoise, 2u);

        delete[] inputDataCovar;
        delete[] inputData;
        delete[] inputDataNoise;
        delete[] measureData;
        delete[] measureDataNoise;
    }

    void Evolution(Data* inputData_input, Parameters* inputParameters_input) override {
        inputData_input[0u][0u] += inputData_input[1u][0u] * deltaTime;
        inputData_input[0u][1u] += inputData_input[1u][1u] * deltaTime;
        inputData_input[0u][2u] += inputData_input[1u][2u] * deltaTime;

        inputData_input[1u][0u] += inputData_input[2u][0u] * deltaTime;
        inputData_input[1u][1u] += inputData_input[2u][1u] * deltaTime;
        inputData_input[1u][2u] += inputData_input[2u][2u] * deltaTime;
    }

    void Observation(Data* inputData_input, Parameters* inputParameters_input, Data* observationData_output) override {
        observationData_output[0u][0u] = inputData_input[1u][0u];
        observationData_output[0u][1u] = inputData_input[1u][1u];
        observationData_output[0u][2u] = inputData_input[1u][2u];

        observationData_output[1u][0u] = inputData_input[2u][0u];
        observationData_output[1u][1u] = inputData_input[2u][1u];
        observationData_output[1u][2u] = inputData_input[2u][2u];
    }
};

class GravitationOutput : public Output
{

};

class HeatFluxEstimation : public Input
{
private:
    double* temperature;
    double* heatFlux;

    double* temperatureCovar;
    double* heatFluxCovar;

    double* temperatureNoise;
    double* heatFluxNoise;

    double* measuredTemperature;
    double* measuredTemperatureNoise;

    long long unsigned* subdivisions;

    double* delta;

    // Encoding the thermal properties of the solid [W / m K]
    double K(double T){
        return 12.45 + 0.014*T + 2.517*T*T*0.000001;
    }

    // Encoding the thermal properties of the solid [J / m^3 K]
    double C(double T){
        return 1324.75 * T + 3557900;
    }

protected:

public:
    HeatFluxEstimation(double* temperature_input, double* heatFlux_input, double* temperatureCovar_input, double* heatFluxCovar_input, double* temperatureNoise_input, double* heatFluxNoise_input, double* measuredTemperature_input, double* measuredTemperatureNoise_input, double Lx, double Ly, double Lz, double Lt, long long unsigned subdivision_X, long long unsigned subdivision_Y, long long unsigned subdivision_Z, long long unsigned subdivision_T){
        subdivisions = new long long unsigned[4u];
        subdivisions[0u] = subdivision_X;
        subdivisions[1u] = subdivision_Y;
        subdivisions[2u] = subdivision_Z;
        subdivisions[3u] = subdivision_T;
        delta = new double[4u];
        delta[0u] = Lx/subdivision_X;
        delta[1u] = Ly/subdivision_Y;
        delta[2u] = Lz/subdivision_Z;
        delta[3u] = Lt/subdivision_T;
        long long unsigned Sxy = subdivision_X * subdivision_Y;
        long long unsigned Sxyz = Sxy * subdivision_Z;
        temperature = new double[Sxyz];
        heatFlux = new double[Sxy];
        temperatureCovar = new double[Sxyz];
        heatFluxCovar = new double[Sxy];
        temperatureNoise = new double[Sxyz];
        heatFluxNoise = new double[Sxy];
        measuredTemperature = new double[Sxy];
        measuredTemperatureNoise = new double[Sxy];
        for(unsigned i = 0u; i < Sxyz; i++){
            temperature[i] = temperature_input[i];
        }
        for(unsigned i = 0u; i < Sxy; i++){
            heatFlux[i] = heatFlux_input[i];
        }
        for(unsigned i = 0u; i < Sxyz; i++){
            temperatureCovar[i] = temperatureCovar_input[i];
        }
        for(unsigned i = 0u; i < Sxy; i++){
            heatFluxCovar[i] = heatFluxCovar_input[i];
        }
        for(unsigned i = 0u; i < Sxyz; i++){
            temperatureNoise[i] = temperatureNoise_input[i];
        }
        for(unsigned i = 0u; i < Sxy; i++){
            heatFluxNoise[i] = heatFluxNoise_input[i];
        }
        for(unsigned i = 0u; i < Sxyz; i++){
            measuredTemperature[i] = measuredTemperature_input[i];
        }
        for(unsigned i = 0u; i < Sxy; i++){
            measuredTemperatureNoise[i] = measuredTemperatureNoise_input[i];
        }
    }

    ~HeatFluxEstimation(){
        delete[] measuredTemperatureNoise;
        delete[] measuredTemperature;
        delete[] heatFluxNoise;
        delete[] temperatureNoise;
        delete[] heatFluxCovar;
        delete[] temperatureCovar;
        delete[] heatFlux;
        delete[] temperature;
        delete[] delta;
        delete[] subdivisions;
    }

    void LoadInput() {
        long long unsigned Sxy = subdivisions[0u] * subdivisions[1u];
        long long unsigned Sxyz = Sxy * subdivisions[2u];

        Parameters* inputParameters = new Parameters[2u];
        inputParameters[0u] = Parameters("Subdivisions", subdivisions,4u,sizeof(long long unsigned));
        inputParameters[1u] = Parameters("Delta Values",delta,4u,sizeof(double));

        Data* inputData = new Data[2u];
        inputData[0u] = Data("Temperature",temperature,Sxyz);
        inputData[1u] = Data("Heat Flux",heatFlux,Sxy);
        Data* inputDataCovariance = new Data[2u];
        inputDataCovariance[0u] = Data("Temperature Covariance",temperatureCovar,Sxyz);
        inputDataCovariance[1u] = Data("Heat Flux Covariance",heatFluxCovar,Sxy);
        Data* inputDataNoise = new Data[2u];
        inputDataNoise[0u] = Data("Temperature Noise",temperatureNoise,Sxyz);
        inputDataNoise[1u] = Data("Heat Flux Noise",heatFluxNoise,Sxyz);

        Data* measureData = new Data[1u];
        measureData[0u] = Data("Measured Temperature",temperature,Sxy);
        Data* measureDataNoise = new Data[1u];
        measureDataNoise[0u] = Data("Measured Temperature Noise",temperatureNoise,Sxy);

        Initialize(inputData,inputDataCovariance,inputDataNoise,2u,inputParameters,2u,measureData,measureDataNoise,1u);

        delete[] measureDataNoise;
        delete[] measureData;
        delete[] inputDataNoise;
        delete[] inputDataCovariance;
        delete[] inputData;
        delete[] inputParameters;
    }

    double Differential(double temperature_in_pos, double temperature_in, double temperature_in_neg, double size){
        double positive = (2.0*K(temperature_in_pos)*K(temperature_in))/(K(temperature_in_pos) + K(temperature_in));
        positive *= (temperature_in_pos-temperature_in);
        double negative = (2.0*K(temperature_in_neg)*K(temperature_in))/(K(temperature_in_neg) + K(temperature_in));
        negative *= (temperature_in_neg-temperature_in);
        return (positive + negative)/(size*size);
    }

    void Evolution(Data* inputData_input, Parameters* inputParameters_input) override {
        unsigned Sx = inputParameters_input[0u].GetValue<long long unsigned>(0u);
        unsigned Sy = inputParameters_input[0u].GetValue<long long unsigned>(1u);
        unsigned Sz = inputParameters_input[0u].GetValue<long long unsigned>(2u);
        unsigned Sxy = Sx*Sy;
        unsigned Sxyz = Sxy*Sz;
        double dx = inputParameters_input[1u].GetValue<double>(0u);
        double dy = inputParameters_input[1u].GetValue<double>(1u);
        double dz = inputParameters_input[1u].GetValue<double>(2u);
        double dt = inputParameters_input[1u].GetValue<double>(3u);
        double diffX, diffY, diffZ;
        double auxPos, aux, auxNeg;
        unsigned index;
        for(unsigned k = 0u; k < Sz; k++){
            for(unsigned j = 0u; j < Sy; j++){
                for(unsigned i = 0u; i < Sx; i++){
                    index = (k)*Sxy+(j)*Sx+(i);
                    aux = inputData_input[0u][index];
                    auxPos = (i < Sx-1u) ? inputData_input[0u][index+1u] : aux;
                    auxNeg = (i > 1u) ? inputData_input[0u][index-1u] : aux;
                    diffX = Differential(auxPos,aux,auxNeg,Sx);
                    auxPos = (j < Sx-1u) ? inputData_input[0u][index+Sx] : aux;
                    auxNeg = (j > 1u) ? inputData_input[0u][index-Sx] : aux;
                    diffY = Differential(auxPos,aux,auxNeg,Sy);
                    auxPos = (k < Sx-1u) ? inputData_input[0u][index+Sxy] : aux;
                    auxNeg = (k > 1u) ? inputData_input[0u][index-Sxy] : aux;
                    diffZ = Differential(auxPos,aux,auxNeg,Sz);
                    inputData_input[0u][index] += (diffX+diffY+diffZ)*(dt/C(aux));
                }
            }
        }
    }

    void Observation(Data* inputData_input, Parameters* inputParameters_input, Data* observationData_output) override {
        unsigned Sx = inputParameters_input[0u].GetValue<long long unsigned>(0u);
        unsigned Sy = inputParameters_input[0u].GetValue<long long unsigned>(1u);
        for(unsigned j = 0u; j < Sy; j++){
            for(unsigned i = 0u; i < Sx; i++){
                observationData_output[0u][(j)*Sx+(i)] = inputData_input[0u][(j)*Sx+(i)];
            }
        }
    }
};

int main(){
    std::cout << "\nStart Execution\n\n";
    GravitationInput* input = new(std::nothrow) GravitationInput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 1.0);
    GravitationOutput* output = new(std::nothrow) GravitationOutput();
    
    input->LoadInput();
    unsigned L = 3u;

    UKF* UKFMod = new(std::nothrow) UKF();

    UKFMod->Initialize(input,output);

    UKFMod->Solve();


    //Data* data = state->GetPoint()->GetData();
    //Data* dataCovar = state->GetPointCovariance()->GetDataCovariance();
    //for(unsigned i = 0; i < state->GetPoint()->GetLengthData(); i++){
    //    data[i].print();
    //    dataCovar[i].print();
    //}

    delete UKFMod;
    //delete state;
    delete input;

    std::cout << "\nEnd Execution\n";
    return 0;
}