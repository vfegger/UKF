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


int main(){
    std::cout << "\nStart Execution\n\n";
    GravitationInput* test = new(std::nothrow) GravitationInput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 1.0);
    
    test->LoadInput();
    unsigned L = 3u;

    UKF* UKFMod = new(std::nothrow) UKF();

    UKFMod->Initialize(test);


    UKFMod->Solve();
    
    //Data* data = state->GetPoint()->GetData();
    //Data* dataCovar = state->GetPointCovariance()->GetDataCovariance();
    //for(unsigned i = 0; i < state->GetPoint()->GetLengthData(); i++){
    //    data[i].print();
    //    dataCovar[i].print();
    //}

    delete UKFMod;
    //delete state;
    delete test;

    std::cout << "\nEnd Execution\n";
    return 0;
}