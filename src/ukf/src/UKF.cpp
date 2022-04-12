#include "../include/UKF.hpp"


UKF::UKF(UKFMemory* memory_in, double alpha, double beta, double kappa){
    memory = memory_in;
    bool isValid =
    memory->GetState()->GetValidation() &&
    memory->GetStateCovariance()->GetValidation() &&
    memory->GetStateNoise()->GetValidation() &&
    memory->GetMeasure()->GetValidation() &&
    memory->GetMeasureNoise()->GetValidation();
    if(!isValid){
        std::cout << "Error: Memory is not properly initialized.\n";
    }
    alpha = alpha;
    beta = beta;
    kappa = kappa;
    unsigned L = memory->GetState()->GetLength();
    lambda = alpha*alpha*(L+kappa)-L;
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
    
    std::cout << "Generate Sigma Points based on Cholesky Decompostion\n";
    unsigned sigmaPointsLength = 2u*lengthState + 1u;
    double* WeightMean = new(std::nothrow) double[sigmaPointsLength];
    double* WeightCovariance = new(std::nothrow) double[sigmaPointsLength];
    for(unsigned i = 0; i < sigmaPointsLength; i++){
        WeightMean[i] = (i == 0) ? lambda/(lengthState+lambda) : 0.5/(lengthState+lambda);
        WeightCovariance[i] = WeightMean[i] - (i == 0) ? (1-alpha*alpha-beta) : 0.0;
    }
    Data* sigmaPointsState = new(std::nothrow) Data[sigmaPointsLength];
    Data::InstantiateMultiple(sigmaPointsState,*state,sigmaPointsLength);
    Data* sigmaPointsObservation = new(std::nothrow) Data[sigmaPointsLength];
    Data::InstantiateMultiple(sigmaPointsObservation,*measure,sigmaPointsLength);
    Math::DistributeOperation(Math::AddInPlace,sigmaPointsState->GetPointer(), chol, lengthState, lengthState, lengthState, lengthState,lengthState,0u);
    Math::DistributeOperation(Math::SubInPlace,sigmaPointsState->GetPointer(), chol, lengthState, lengthState, lengthState, lengthState,(lengthState+1u)*lengthState,0u);
    delete[] chol;
    std::cout << "Evolution Step\n";
    for(unsigned i = 0u; i < sigmaPointsLength; i++){
        memory->Evolution(sigmaPointsState[i], *parameter);
    }
    
    std::cout << "Observation Step\n";
    for(unsigned i = 0u; i < sigmaPointsLength; i++){
        memory->Observation(sigmaPointsState[i], *parameter, sigmaPointsObservation[i]);
    }
    
    std::cout << "Mean\n";
    Math::Mean(statePointer, sigmaPointsState->GetPointer(), lengthState, sigmaPointsLength, WeightMean);
    Math::Mean(observationPointer, sigmaPointsObservation->GetPointer(), lengthObservation, sigmaPointsLength, WeightMean);
    
    std::cout << "Covariance\n";
    Math::DistributeOperation(Math::SubInPlace,sigmaPointsState->GetPointer(), statePointer, lengthState, lengthState, 0u, sigmaPointsLength);
    Math::DistributeOperation(Math::SubInPlace,sigmaPointsObservation->GetPointer(), observationPointer, lengthObservation, lengthObservation, 0u, sigmaPointsLength);
    Math::MatrixMultiplication(stateCovariancePointer,
    sigmaPointsState->GetPointer(), Math::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    sigmaPointsState->GetPointer(), Math::MatrixStructure::Transposed, lengthState, sigmaPointsLength,
    WeightCovariance);
    Math::AddInPlace(stateCovariancePointer,stateNoisePointer,lengthState*lengthState);
    Math::MatrixMultiplication(observationCovariancePointer,
    sigmaPointsObservation->GetPointer(), Math::MatrixStructure::Natural, lengthObservation, sigmaPointsLength,
    sigmaPointsObservation->GetPointer(), Math::MatrixStructure::Transposed, lengthObservation, sigmaPointsLength,
    WeightCovariance);
    Math::AddInPlace(observationCovariancePointer,measureNoisePointer,lengthObservation*lengthObservation);
    Math::MatrixMultiplication(crossCovariancePointer,
    sigmaPointsState->GetPointer(), Math::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    sigmaPointsObservation->GetPointer(), Math::MatrixStructure::Transposed, lengthObservation, sigmaPointsLength,
    WeightCovariance);
    //  TODO: RHSolver to find K = Pxy*(Pyy^-1) <=> K * Pyy = Pxy
    Math::RHSolver(kalmanGainPointer,observationCovariancePointer,crossCovariancePointer,lengthState,lengthObservation);
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

    std::cout << "Delete Auxiliary Structures\n";
    delete[] WeightMean;
    delete[] WeightCovariance;

    delete[] crossCovariancePointer;
    delete[] kalmanGainPointer;
    delete[] inverseObservationCovariancePointer;

    delete[] sigmaPointsState;
    delete[] sigmaPointsObservation;

    delete observation;
    delete observationCovariance;

    std::cout << "Iteration Ended\n";
}