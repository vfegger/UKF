#include "UKF.hpp"

UKF::UKF(){

}

UKF::~UKF(){

}

void UKF::Initialize(Input* input_in){
    input = input_in;
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