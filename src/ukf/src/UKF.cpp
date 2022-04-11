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

    Data* observation = new Data(*measure);
    double* observationPointer = state->GetPointer();
    Data* observationCovariance = new Data(*stateCovariance);
    double* observationCovariancePointer = state->GetPointer();


    unsigned lengthState = state->GetLength();
    unsigned lengthObservation = observation->GetLength();

    //Methods
    //  Calculate Cholesky Decomposition
    double* chol = new(std::nothrow) double[lengthState * lengthState];
    Math::CholeskyDecomposition(chol,stateCovariancePointer,lengthState,lengthState);
    delete[] chol;
    //  Generate Sigma Points based on Cholesky Decompostion
    unsigned sigmaPointsLength = 2u*lengthState + 1u;
    Data* sigmaPointsState = new(std::nothrow) Data[sigmaPointsLength];
    Data::InstantiateMultiple(sigmaPointsState,*state,sigmaPointsLength);
    Data* sigmaPointsObservation = new(std::nothrow) Data[sigmaPointsLength];
    Data::InstantiateMultiple(sigmaPointsObservation,*measure,sigmaPointsLength);
    //  Evolution Step
    for(unsigned i = 0u; i < sigmaPointsLength; i++){
        memory->Evolution(sigmaPointsState[i], *parameter);
    }
    //  Observation Step
    for(unsigned i = 0u; i < sigmaPointsLength; i++){
        memory->Observation(sigmaPointsState[i], *parameter, sigmaPointsObservation[i]);
    }
    //  Mean
    Math::Mean(statePointer, sigmaPointsState->GetPointer(), lengthState, sigmaPointsLength);
    Math::Mean(observationPointer, sigmaPointsObservation->GetPointer(), lengthObservation, sigmaPointsLength);
    //  Covariance
    Math::DistributeOperation(Math::SubInPlace,sigmaPointsState->GetPointer(), statePointer, lengthState, lengthState, 0u, sigmaPointsLength);
    Math::DistributeOperation(Math::SubInPlace,sigmaPointsObservation->GetPointer(), observationPointer, lengthObservation, lengthObservation, 0u, sigmaPointsLength);
    Math::MatrixMultiplication(stateCovariancePointer,
    sigmaPointsState->GetPointer(), Math::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    sigmaPointsState->GetPointer(), Math::MatrixStructure::Transposed, lengthState, sigmaPointsLength);
    Math::MatrixMultiplication(observationCovariancePointer,
    sigmaPointsObservation->GetPointer(), Math::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    sigmaPointsObservation->GetPointer(), Math::MatrixStructure::Transposed, lengthState, sigmaPointsLength);
    Data* crossCovariance = new Data(*stateCovariance);
    double* crossCovariancePointer = crossCovariance->GetPointer();
    Math::MatrixMultiplication(crossCovariancePointer,
    sigmaPointsState->GetPointer(), Math::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    sigmaPointsObservation->GetPointer(), Math::MatrixStructure::Transposed, lengthState, sigmaPointsLength);
    //  TODO: RHSolver to find K = Pxy*(Pyy^-1) <=> K * Pyy = Pxy
    Data* inverseObservationCovariance = new Data(*observationCovariance);
    double* inverseObservationCovariancePointer = inverseObservationCovariance->GetPointer();
    //  Kalman Gain Calulation
    Data* kalmanGain = new Data(*crossCovariance);
    double* kalmanGainPointer = kalmanGain->GetPointer();
    Math::MatrixMultiplication(kalmanGainPointer,
    crossCovariancePointer, Math::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    inverseObservationCovariancePointer, Math::MatrixStructure::Transposed, lengthState, sigmaPointsLength);
    //  State Update
    //      Mean
    Math::SubInPlace(measurePointer,observationPointer,lengthObservation);
    Math::MatrixMultiplication(observationPointer,
    kalmanGainPointer, Math::MatrixStructure::Natural, lengthState, lengthObservation,
    measurePointer, Math::MatrixStructure::Natural, lengthObservation, 1);
    //      Covariance
    Math::MatrixMultiplication(crossCovariancePointer,
    kalmanGainPointer, Math::MatrixStructure::Natural, lengthState, lengthObservation,
    observationCovariancePointer, Math::MatrixStructure::Natural, lengthObservation, lengthObservation);
    Math::MatrixMultiplication(observationCovariancePointer,
    crossCovariancePointer, Math::MatrixStructure::Natural, lengthState, lengthObservation,
    kalmanGainPointer, Math::MatrixStructure::Transposed, lengthState, lengthObservation);
    Math::AddInPlace(stateCovariancePointer, observationCovariancePointer, lengthState*lengthState);
    
}