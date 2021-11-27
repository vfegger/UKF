#include "UKF.hpp"
#include <new>

UKF::UKF(){

}

UKF::~UKF(){

}

void UKF::Initialize(Input* input_in){
    input = input_in;
}

void UKF::SigmaPointsGenerator(State* state, Point* &sigmaPoints, unsigned &sigmaLength){
    unsigned stateLength = state->GetStateLength();
    sigmaLength = 2u*stateLength+1u;
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
    Parameters* parameters = input->GetParameters();
    State* state = input->GetState();
    unsigned length = state->GetStateLength();
    unsigned sigmaLength = 0u;
    // Generate Sigma Points of the state
    Point* sigmaPoints;
    SigmaPointsGenerator(state,sigmaPoints,sigmaLength);
    // Evolve all Sigma Points of the state
    for(unsigned i = 0; i < sigmaLength; i++){
        sigmaPoints[i].UpdateDataFromArray();
        input->Evolution(sigmaPoints[i].GetData(),parameters);
        sigmaPoints[i].UpdateArrayFromData();
    }
    // Get Mean of sigma points of the state
    double* mean = state->GetPoint()->GetState();
    for(unsigned i = 0; i < length; i++){
            mean[i] = 0;
        }
    for(unsigned k = 0; k < sigmaLength; k++){
        double* aux = sigmaPoints[k].GetState();
        for(unsigned i = 0; i < length; i++){
            mean[i] += aux[i];
        }
    }
    // Calculate the covariance of the state
    double* covariance = state->GetPointCovariance()->GetStateCovariance();
    for(unsigned i = 0u; i < length; i++){
        for(unsigned j = 0u; j < length; j++){
            covariance[i*length+j] = 0.0;
        }
    }
    for(unsigned k = 0; k < sigmaLength; k++){
        double* aux = sigmaPoints[k].GetState();
        for(unsigned i = 0u; i < length; i++){
            for(unsigned j = 0; j < length; j++){
                covariance[i*length+j] += (aux[i]-mean[i])*(aux[j]-mean[j]);
            }
        }
    }
    // Observe all Sigma Points of the state
    for(unsigned i = 0; i < sigmaLength; i++){
        sigmaPoints[i].UpdateDataFromArray();
        //input->Observation(sigmaPoints[i].GetData(),parameters,sigmaPointsObservation);
        //sigmaPointsObservation[i].UpdateArrayFromData();
    }
    // Get Mean of observation of all sigma Points
    for(unsigned i = 0; i < sigmaLength; i++){
        sigmaPoints[i].UpdateDataFromArray();
        //input->Observation(sigmaPoints[i].GetData(),parameters,sigmaPointsObservation);
        //sigmaPointsObservation[i].UpdateArrayFromData();
    }
    // Calculate covariance of observation of the state

    // Calculate cross-covariance between the state and the observation

    // Calculate the Kalman Gain

    // Update state and Covariance
}

void UKF::Export(Output* output_out){
    output_out = output;
}