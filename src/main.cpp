#include <iostream>
#include "Input.hpp"
#include "Output.hpp"
#include "UKF.hpp"

class GravitationInput : public Input
{
private:
    double* position;
    double* velocity;
    double* acceleration;

    double* positionCovar;
    double* velocityCovar;
    double* accelerationCovar;

    double deltaTime;
public:
    GravitationInput(double px, double py, double pz, double vx, double vy, double vz, double ax, double ay, double az){
        //State
        position = new double[3u];
        position[0u] = px;
        position[1u] = py;
        position[2u] = pz;

        velocity = new double[3u];
        velocity[0u] = vx;
        velocity[1u] = vy;
        velocity[2u] = vz;

        acceleration = new double[3u];
        acceleration[0u] = ax;
        acceleration[1u] = ay;
        acceleration[2u] = az;

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

    }

    ~GravitationInput(){
        delete[] accelerationCovar;
        delete[] velocityCovar;
        delete[] positionCovar;
        delete[] acceleration;
        delete[] velocity;
        delete[] position;
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
        
        Initialize(inputData, inputDataCovar, 3u, NULL, 0);
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
        observationData_output[1u][0u] = inputData_input[1u][0u];
        observationData_output[1u][2u] = inputData_input[1u][2u];
        observationData_output[1u][1u] = inputData_input[1u][1u];

        observationData_output[2u][0u] = inputData_input[2u][0u];
        observationData_output[2u][1u] = inputData_input[2u][1u];
        observationData_output[2u][2u] = inputData_input[2u][2u];
    }
};


int main(){
    std::cout << "\nStart Execution\n\n";
    GravitationInput* test = new GravitationInput(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, -10.0, 0.0
    );
    
    test->LoadInput();
    unsigned L = 3u;
    Data* data = test->GetState()->GetPoint()->GetData();
    Data* dataCovar = test->GetState()->GetPointCovariance()->GetDataCovariance();
    for(unsigned i = 0u; i < L; i++){
        data[i].print();
        dataCovar[i].print();
    }

    UKF* UKFMod = new UKF();

    UKFMod->Initialize(test);

    UKFMod->Solve();

    delete UKFMod;
    delete test;

    std::cout << "\nEnd Execution\n";
    return 0;
}