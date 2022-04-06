#include "../include/UKF.hpp"

//UKF Iteration
void UKF::Iterate(){
    //Initialization of variables
    Parameter* parameter = memory->GetParameter();

    Data* state = memory->GetState();
    double* statePointer = state->GetPointer();
    Data* stateCovariance = memory->GetStateCovariance();
    double* stateCovariancePointer = state->GetPointer();
    Data* stateNoise = memory->GetStateNoise();
    double* stateNoisePointer = state->GetPointer();

    Data* measure = memory->GetMeasure();
    double* measurePointer = state->GetPointer();
    Data* measureNoise = memory->GetMeasureNoise();
    double* measureNoisePointer = state->GetPointer();

    //Methods
    //Cholesky(chol,stateCovariancePointer);
    //GenerateSigmaPoints(sigmaPoints,statePointer,chol);
    //memory->Evolution(sigmaPoints,*parameter);
    //Mean(state,sigmaPoints,weightMean);
    //Covariance(stateCovariance,sigmaPoints-state,weightCovar);
    //memory->Observation(sigmaPoints,*parameter,sigmaObservations);
    //Mean(observation,sigmaObservations,weightMean);
    //Covariance(observationCovariance,sigmaObservation-observation,weightCovar);
    //CrossCovariance(crossCovariance,sigmaPoints-state,sigmaObservation-observation);
    //RHSolver(K,crossCovariance,observationCovariance);
    //UpdateState(state,K,measure,observation);
    //UpdateCovariance(stateCovariance,K,observationCovariance,KT);

    
}