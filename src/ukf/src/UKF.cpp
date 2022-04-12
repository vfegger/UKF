#include "../include/UKF.hpp"


UKF::UKF(UKFMemory* memory_in){
    memory = memory_in;
}
//UKF Iteration
void UKF::Iterate(){
    //Initialization of variables
    Parameter* parameter = memory->GetParameter();

    Data* state = memory->GetState();
    double* statePointer = state->GetPointer();
    DataCovariance* stateCovariance = memory->GetStateCovariance();
    double* stateCovariancePointer = stateCovariance->GetPointer();
    DataCovariance* stateNoise = memory->GetStateNoise();
    double* stateNoisePointer = stateNoise->GetPointer();

    Data* measure = memory->GetMeasure();
    double* measurePointer = measure->GetPointer();
    DataCovariance* measureNoise = memory->GetMeasureNoise();
    double* measureNoisePointer = measureNoise->GetPointer();

    Data* observation = new Data(*measure);
    double* observationPointer = observation->GetPointer();
    DataCovariance* observationCovariance = new DataCovariance(*observation);
    double* observationCovariancePointer = observationCovariance->GetPointer();


    unsigned lengthState = state->GetLength();
    unsigned lengthObservation = observation->GetLength();

    std::cout << "State length: " << lengthState << "; Observation length: " << lengthObservation << "\n";

    double* crossCovariancePointer = new(std::nothrow) double[lengthState * lengthObservation];
    double* inverseObservationCovariancePointer = new(std::nothrow) double[lengthObservation*lengthObservation];
    double* kalmanGainPointer = new(std::nothrow) double[lengthState*lengthObservation];
    for(unsigned i = 0u; i < lengthState * lengthObservation; i++){
        crossCovariancePointer[i] = 0.0;
        kalmanGainPointer[i] = 0.0;
    }
    for(unsigned i = 0u; i < lengthObservation * lengthObservation; i++){
        inverseObservationCovariancePointer[i] = 0.0;
    }
    //Methods
    std::cout << "Calculate Cholesky Decomposition\n";
    double* chol = new(std::nothrow) double[lengthState * lengthState];
    Math::CholeskyDecomposition(chol,stateCovariancePointer,lengthState,lengthState);
    delete[] chol;
    
    std::cout << "Generate Sigma Points based on Cholesky Decompostion\n";
    unsigned sigmaPointsLength = 2u*lengthState + 1u;
    Data* sigmaPointsState = new(std::nothrow) Data[sigmaPointsLength];
    Data::InstantiateMultiple(sigmaPointsState,*state,sigmaPointsLength);
    Data* sigmaPointsObservation = new(std::nothrow) Data[sigmaPointsLength];
    Data::InstantiateMultiple(sigmaPointsObservation,*measure,sigmaPointsLength);
    
    std::cout << "Evolution Step\n";
    for(unsigned i = 0u; i < sigmaPointsLength; i++){
        memory->Evolution(sigmaPointsState[i], *parameter);
    }
    
    std::cout << "Observation Step\n";
    for(unsigned i = 0u; i < sigmaPointsLength; i++){
        memory->Observation(sigmaPointsState[i], *parameter, sigmaPointsObservation[i]);
    }
    
    std::cout << "Mean\n";
    Math::Mean(statePointer, sigmaPointsState->GetPointer(), lengthState, sigmaPointsLength);
    Math::Mean(observationPointer, sigmaPointsObservation->GetPointer(), lengthObservation, sigmaPointsLength);
    
    std::cout << "Covariance\n";
    Math::DistributeOperation(Math::SubInPlace,sigmaPointsState->GetPointer(), statePointer, lengthState, lengthState, 0u, sigmaPointsLength);
    Math::DistributeOperation(Math::SubInPlace,sigmaPointsObservation->GetPointer(), observationPointer, lengthObservation, lengthObservation, 0u, sigmaPointsLength);
    Math::MatrixMultiplication(stateCovariancePointer,
    sigmaPointsState->GetPointer(), Math::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    sigmaPointsState->GetPointer(), Math::MatrixStructure::Transposed, lengthState, sigmaPointsLength);
    Math::MatrixMultiplication(observationCovariancePointer,
    sigmaPointsObservation->GetPointer(), Math::MatrixStructure::Natural, lengthObservation, sigmaPointsLength,
    sigmaPointsObservation->GetPointer(), Math::MatrixStructure::Transposed, lengthObservation, sigmaPointsLength);
    Math::MatrixMultiplication(crossCovariancePointer,
    sigmaPointsState->GetPointer(), Math::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    sigmaPointsObservation->GetPointer(), Math::MatrixStructure::Transposed, lengthObservation, sigmaPointsLength);
    //  TODO: RHSolver to find K = Pxy*(Pyy^-1) <=> K * Pyy = Pxy
    std::cout << "Kalman Gain Calulation\n";
    Math::MatrixMultiplication(kalmanGainPointer,
    crossCovariancePointer, Math::MatrixStructure::Natural, lengthState, lengthObservation,
    inverseObservationCovariancePointer, Math::MatrixStructure::Transposed, lengthObservation, lengthObservation);
    
    std::cout << "State Update\n";
    std::cout << "State\n";
    Math::SubInPlace(measurePointer,observationPointer,lengthObservation);
    Math::MatrixMultiplication(observationPointer,
    kalmanGainPointer, Math::MatrixStructure::Natural, lengthState, lengthObservation,
    measurePointer, Math::MatrixStructure::Natural, lengthObservation, 1u);
    std::cout << "State Covariance\n";
    Math::MatrixMultiplication(crossCovariancePointer,
    kalmanGainPointer, Math::MatrixStructure::Natural, lengthState, lengthObservation,
    observationCovariancePointer, Math::MatrixStructure::Natural, lengthObservation, lengthObservation);
    Math::MatrixMultiplication(observationCovariancePointer,
    crossCovariancePointer, Math::MatrixStructure::Natural, lengthState, lengthObservation,
    kalmanGainPointer, Math::MatrixStructure::Transposed, lengthState, lengthObservation);
    Math::AddInPlace(stateCovariancePointer, observationCovariancePointer, lengthState*lengthState);

    delete[] crossCovariancePointer;
    delete[] kalmanGainPointer;
    delete[] inverseObservationCovariancePointer;

    std::cout << "Iteration Ended\n";
}