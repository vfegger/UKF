#include "../include/UKF.hpp"

#ifdef _OPENMP
namespace MathUKF = MathOpenMP;
#else
namespace MathUKF = Math;
#endif

UKF::UKF(UKFMemory* memory_in, double alpha_in, double beta_in, double kappa_in){
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
    alpha = alpha_in;
    beta = beta_in;
    kappa = kappa_in;
    unsigned L = memory->GetState()->GetLength();
    lambda = alpha*alpha*(L+kappa)-L;
}
//UKF Iteration
void UKF::Iterate(Timer& timer){
    timer.Reset();
    timer.Start();
    //Initialization of variables
    Parameter* parameter = memory->GetParameter();

    Data* state = memory->GetState();
    double* statePointer = state->GetPointer();
    DataCovariance* stateCovariance = memory->GetStateCovariance();
    double* stateCovariancePointer = stateCovariance->GetPointer();
    DataCovariance* stateNoise = memory->GetStateNoise();
    double* stateNoisePointer = stateNoise->GetPointer();

    Data* measure = new Data(*(memory->GetMeasure()));
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
    double* kalmanGainPointer = new(std::nothrow) double[lengthState*lengthObservation];
    for(unsigned i = 0u; i < lengthState * lengthObservation; i++){
        crossCovariancePointer[i] = 0.0;
        kalmanGainPointer[i] = 0.0;
    }
    timer.Save();

    //Methods
    std::cout << "Calculate Cholesky Decomposition\n";
    double* chol = new(std::nothrow) double[lengthState * lengthState];
    for(unsigned i = 0u; i < lengthState*lengthState; i++){
        chol[i] = 0.0;
    }
    MathUKF::ConstantMultiplicationInPlace(stateCovariancePointer,lengthState+lambda,lengthState*lengthState);
    MathUKF::CholeskyDecomposition(chol,stateCovariancePointer,lengthState,lengthState);
    timer.Save();

    std::cout << "Generate Sigma Points based on Cholesky Decompostion\n";
    unsigned sigmaPointsLength = 2u*lengthState + 1u;
    double* WeightMean = new(std::nothrow) double[sigmaPointsLength];
    double* WeightCovariance = new(std::nothrow) double[sigmaPointsLength];
    for(unsigned i = 0; i < sigmaPointsLength; i++){
        if(i == 0u){
            WeightMean[i] = lambda/(lengthState+lambda);
            WeightCovariance[i] = lambda/(lengthState+lambda) + (1.0-alpha*alpha+beta);
        } else {
            WeightMean[i] = 0.5/(lengthState+lambda);
            WeightCovariance[i] = 0.5/(lengthState+lambda);
        }
    }
    Data* sigmaPointsState = NULL;
    Data* sigmaPointsObservation = NULL;
    Data::InstantiateMultiple(sigmaPointsState,*state,sigmaPointsLength);
    Data::InstantiateMultiple(sigmaPointsObservation,*measure,sigmaPointsLength);
    MathUKF::DistributeOperation(MathUKF::AddInPlace,sigmaPointsState->GetPointer(), chol, lengthState, lengthState, lengthState, lengthState, lengthState, 0u);
    MathUKF::DistributeOperation(MathUKF::SubInPlace,sigmaPointsState->GetPointer(), chol, lengthState, lengthState, lengthState, lengthState, (lengthState+1u)*lengthState, 0u);
    delete[] chol;
    timer.Save();

    std::cout << "Observation Step\n";
    for(unsigned i = 0u; i < sigmaPointsLength; i++){
        memory->Observation(sigmaPointsState[i], *parameter, sigmaPointsObservation[i]);
    }
    timer.Save();

    std::cout << "Evolution Step\n";
    for(unsigned i = 0u; i < sigmaPointsLength; i++){
        memory->Evolution(sigmaPointsState[i], *parameter);
    }
    timer.Save();
    
    std::cout << "Mean\n";
    MathUKF::Mean(statePointer, sigmaPointsState->GetPointer(), lengthState, sigmaPointsLength, WeightMean);
    MathUKF::Mean(observationPointer, sigmaPointsObservation->GetPointer(), lengthObservation, sigmaPointsLength, WeightMean);
    timer.Save();

    std::cout << "Covariance\n";
    MathUKF::DistributeOperation(MathUKF::SubInPlace,sigmaPointsState->GetPointer(), statePointer, lengthState, lengthState, 0u, sigmaPointsLength);
    MathUKF::DistributeOperation(MathUKF::SubInPlace,sigmaPointsObservation->GetPointer(), observationPointer, lengthObservation, lengthObservation, 0u, sigmaPointsLength);
     
    MathUKF::MatrixMultiplication(stateCovariancePointer, 1.0, 0.0,
    sigmaPointsState->GetPointer(), MathUKF::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    sigmaPointsState->GetPointer(), MathUKF::MatrixStructure::Transposed, lengthState, sigmaPointsLength,
    WeightCovariance);
    MathUKF::AddInPlace(stateCovariancePointer,stateNoisePointer,lengthState*lengthState);
    
    MathUKF::MatrixMultiplication(observationCovariancePointer, 1.0, 0.0,
    sigmaPointsObservation->GetPointer(), MathUKF::MatrixStructure::Natural, lengthObservation, sigmaPointsLength,
    sigmaPointsObservation->GetPointer(), MathUKF::MatrixStructure::Transposed, lengthObservation, sigmaPointsLength,
    WeightCovariance);
    MathUKF::AddInPlace(observationCovariancePointer,measureNoisePointer,lengthObservation*lengthObservation);
    
    MathUKF::MatrixMultiplication(crossCovariancePointer, 1.0, 0.0,
    sigmaPointsState->GetPointer(), MathUKF::MatrixStructure::Natural, lengthState, sigmaPointsLength,
    sigmaPointsObservation->GetPointer(), MathUKF::MatrixStructure::Transposed, lengthObservation, sigmaPointsLength,
    WeightCovariance);

    timer.Save();

    //  RHSolver to find K = Pxy*(Pyy^-1) <=> K * Pyy = Pxy
    std::cout << "Kalman Gain Calulation\n";
    MathUKF::RHSolver(kalmanGainPointer,observationCovariancePointer,crossCovariancePointer,lengthState,lengthObservation);
    timer.Save();
    
    std::cout << "State Update\n";
    std::cout << "State\n";
    MathUKF::SubInPlace(measurePointer,observationPointer,lengthObservation);
    MathUKF::MatrixMultiplication(statePointer, 1.0, 1.0,
    kalmanGainPointer, MathUKF::MatrixStructure::Natural, lengthState, lengthObservation,
    measurePointer, MathUKF::MatrixStructure::Natural, lengthObservation, 1u);
    timer.Save();
    
    std::cout << "State Covariance\n";
    MathUKF::MatrixMultiplication(crossCovariancePointer, 1.0, 0.0,
    kalmanGainPointer, MathUKF::MatrixStructure::Natural, lengthState, lengthObservation,
    observationCovariancePointer, MathUKF::MatrixStructure::Natural, lengthObservation, lengthObservation);
    
    MathUKF::MatrixMultiplication(stateCovariancePointer, -1.0, 1.0,
    crossCovariancePointer, MathUKF::MatrixStructure::Natural, lengthState, lengthObservation,
    kalmanGainPointer, MathUKF::MatrixStructure::Transposed, lengthState, lengthObservation);
    timer.Save();

    std::cout << "Delete Auxiliary Structures\n";
    delete[] WeightMean;
    delete[] WeightCovariance;

    delete[] crossCovariancePointer;
    delete[] kalmanGainPointer;

    delete[] sigmaPointsState;
    delete[] sigmaPointsObservation;

    delete measure;
    delete observation;
    delete observationCovariance;
    timer.Save();

    std::cout << "Iteration Ended\n";
}