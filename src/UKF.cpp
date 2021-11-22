#include "UKF.hpp"
#include <new>

UKF::UKF(){

}

UKF::~UKF(){

}

void UKF::Initialize(Input* input_in){
    input = input_in;
}

void UKF::SigmaPointsGenerator(State* state, Point* sigmaPoints){
    unsigned stateLength = state->GetStateLength();
    unsigned sigmaLength = 2u*stateLength+1u;
    Point* mean = state->GetPoint();
    PointCovariance* covariance = state->GetPointCovariance();
    sigmaPoints = new(std::nothrow) Point[sigmaLength];
    //TODO: Cholesky Decomp.
    double* chol = covariance->GetStateCovariance();
    //TODO: Optimize
    sigmaPoints[0] = Point(mean);
    for(unsigned i = 0u; i < stateLength; i++){
        sigmaPoints[i+1u] = Point(mean);
        double* aux = sigmaPoints[i+1u].GetState();
        for(unsigned j = 0u; j < stateLength; j++){
            aux[j] += chol[i*stateLength+j];  
        }
    }
    for(unsigned i = 0u; i < stateLength; i++){
        sigmaPoints[i+stateLength+1u] = Point(mean);
        double* aux = sigmaPoints[i+stateLength+1u].GetState();
        for(unsigned j = 0u; j < stateLength; j++){
            aux[j] -= chol[i*stateLength+j];  
        }
    }
    return;
}

void UKF::Solve(){
    // Generate Sigma Points of the state

    // Evolve all Sigma Points of the state
    
    // Get Mean of sigma points of the state

    // Calculate the covariance of the state

    // Observe all Sigma Points of the state

    // Get Mean of observation of all sigma Points

    // Calculate covariance of observation of the state

    // Calculate cross-covariance between the state and the observation

    // Calculate the Kalman Gain

    // Update state and Covariance
}

void UKF::Export(Output* output_out){
    output_out = output;
}