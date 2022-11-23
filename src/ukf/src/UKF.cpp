#include "../include/UKF.hpp"

UKF::UKF(Pointer<UKFMemory> memory_in, double alpha_in, double beta_in, double kappa_in)
{
    memory = memory_in;
    bool isValid =
        memory.pointer[0u].GetState().pointer[0u].GetValidation() &&
        memory.pointer[0u].GetStateCovariance().pointer[0u].GetValidation() &&
        memory.pointer[0u].GetStateNoise().pointer[0u].GetValidation() &&
        memory.pointer[0u].GetMeasure().pointer[0u].GetValidation() &&
        memory.pointer[0u].GetMeasureNoise().pointer[0u].GetValidation() &&
        memory.pointer[0u].GetParameter().pointer[0u].GetValidation();
    if (!isValid)
    {
        std::cout << "Error: Memory is not properly initialized.\n";
    }
    alpha = alpha_in;
    beta = beta_in;
    kappa = kappa_in;
    unsigned L = memory.pointer[0u].GetState().pointer[0u].GetLength();
    lambda = alpha * alpha * (L + kappa) - L;
}
// UKF Iteration
void UKF::Iterate(Timer &timer)
{
    timer.Reset();
    timer.Start();
    // Initialization of variables
    Pointer<Parameter> parameter = memory.pointer[0u].GetParameter();

    Pointer<Data> state = memory.pointer[0u].GetState();
    Pointer<double> statePointer = state.pointer[0u].GetPointer();
    Pointer<DataCovariance> stateCovariance = memory.pointer[0u].GetStateCovariance();
    Pointer<double> stateCovariancePointer = stateCovariance.pointer[0u].GetPointer();
    Pointer<DataCovariance> stateNoise = memory.pointer[0u].GetStateNoise();
    Pointer<double> stateNoisePointer = stateNoise.pointer[0u].GetPointer();

    Pointer<Data> measure = MemoryHandler::AllocValue<Data, Data>(memory.pointer[0u].GetMeasure().pointer[0u], memory.pointer[0u].GetMeasure().type, memory.pointer[0u].GetMeasure().context);
    Pointer<double> measurePointer = measure.pointer[0u].GetPointer();
    Pointer<DataCovariance> measureNoise = memory.pointer[0u].GetMeasureNoise();
    Pointer<double> measureNoisePointer = measureNoise.pointer[0u].GetPointer();

    Pointer<Data> observation = MemoryHandler::AllocValue<Data, Data>(measure.pointer[0u], measure.type, measure.context);
    Pointer<double> observationPointer = observation.pointer[0u].GetPointer();
    Pointer<DataCovariance> observationCovariance = MemoryHandler::AllocValue<DataCovariance, Data>(observation.pointer[0u], observation.type, observation.context);
    Pointer<double> observationCovariancePointer = observationCovariance.pointer[0u].GetPointer();

    unsigned lengthState = state.pointer[0u].GetLength();
    unsigned lengthObservation = observation.pointer[0u].GetLength();

    PointerType stateType = statePointer.type;
    PointerContext stateContext = statePointer.context;

    std::cout << "State length: " << lengthState << "; Observation length: " << lengthObservation << "\n";

    Pointer<double> crossCovariancePointer = MemoryHandler::Alloc<double>(lengthState * lengthObservation, stateType, stateContext);
    Pointer<double> kalmanGainPointer = MemoryHandler::Alloc<double>(lengthState * lengthObservation, stateType, stateContext);
    MemoryHandler::Set(crossCovariancePointer, 0.0, 0u, lengthState * lengthObservation);
    timer.Save();

    // Methods
    std::cout << "Calculate Cholesky Decomposition\n";
    Pointer<double> chol = MemoryHandler::Alloc<double>(lengthState * lengthState, stateType, stateContext);
    for (unsigned i = 0u; i < lengthState * lengthState; i++)
    {
        chol.pointer[i] = 0.0;
    }
    Math::Mul(stateCovariancePointer, lengthState + lambda, lengthState * lengthState);
    Math::Decomposition(chol, DecompositionType_Cholesky, stateCovariancePointer, lengthState, lengthState);
    timer.Save();

    std::cout << "Generate Sigma Points based on Cholesky Decompostion\n";
    unsigned sigmaPointsLength = 2u * lengthState + 1u;
    Pointer<double> WeightMean = MemoryHandler::Alloc<double>(sigmaPointsLength, stateType, stateContext);
    Pointer<double> WeightCovariance = MemoryHandler::Alloc<double>(sigmaPointsLength, stateType, stateContext);

    MemoryHandler::Set(WeightMean, lambda / (lengthState + lambda), 0u, 1u);
    MemoryHandler::Set(WeightCovariance, lambda / (lengthState + lambda) + (1.0 - alpha * alpha + beta), 0u, 1u);
    MemoryHandler::Set(WeightMean, 0.5 / (lengthState + lambda), 1u, sigmaPointsLength);
    MemoryHandler::Set(WeightCovariance, 0.5 / (lengthState + lambda), 1u, sigmaPointsLength);

    Pointer<Data> sigmaPointsState = Pointer<Data>();       // TODO
    Pointer<Data> sigmaPointsObservation = Pointer<Data>(); // TODO
    Data::InstantiateMultiple(sigmaPointsState, state.pointer[0u], sigmaPointsLength);
    Data::InstantiateMultiple(sigmaPointsObservation, observation.pointer[0u], sigmaPointsLength);
    Math::Operation(Math::Add, sigmaPointsState.pointer[0u].GetPointer(), chol, lengthState, lengthState, lengthState, lengthState, lengthState, 0u);
    Math::Operation(Math::Sub, sigmaPointsState.pointer[0u].GetPointer(), chol, lengthState, lengthState, lengthState, lengthState, (lengthState + 1u) * lengthState, 0u);
    MemoryHandler::Free(chol);
    timer.Save();

    std::cout << "Observation Step\n";
    for (unsigned i = 0u; i < sigmaPointsLength; i++)
    {
        memory.pointer[0u].Observation(sigmaPointsState.pointer[i], parameter.pointer[0u], sigmaPointsObservation.pointer[i]);
    }
    timer.Save();

    std::cout << "Evolution Step\n";
    for (unsigned i = 0u; i < sigmaPointsLength; i++)
    {
        memory.pointer[0u].Evolution(sigmaPointsState.pointer[i], parameter.pointer[0]);
    }
    timer.Save();

    std::cout << "Mean\n";
    Math::Mean(statePointer, sigmaPointsState.pointer[0u].GetPointer(), lengthState, sigmaPointsLength, WeightMean);
    Math::Mean(observationPointer, sigmaPointsObservation.pointer[0u].GetPointer(), lengthObservation, sigmaPointsLength, WeightMean);
    timer.Save();

    std::cout << "Covariance\n";
    Math::Operation(Math::Sub, sigmaPointsState.pointer[0u].GetPointer(), statePointer, lengthState, sigmaPointsLength, lengthState, 0u);
    Math::Operation(Math::Sub, sigmaPointsObservation.pointer[0u].GetPointer(), observationPointer, lengthObservation, sigmaPointsLength, lengthObservation, 0u);

    Math::MatrixMultiplication(1.0,
                               sigmaPointsState.pointer[0u].GetPointer(), MatrixStructure_Natural, lengthState, sigmaPointsLength,
                               sigmaPointsState.pointer[0u].GetPointer(), MatrixStructure_Transposed, lengthState, sigmaPointsLength,
                               0.0, stateCovariancePointer, MatrixStructure_Natural, lengthState, lengthState,
                               WeightCovariance);
    Math::Add(stateCovariancePointer, stateNoisePointer, lengthState * lengthState);

    Math::MatrixMultiplication(1.0,
                               sigmaPointsObservation.pointer[0u].GetPointer(), MatrixStructure_Natural, lengthObservation, sigmaPointsLength,
                               sigmaPointsObservation.pointer[0u].GetPointer(), MatrixStructure_Transposed, lengthObservation, sigmaPointsLength,
                               0.0, observationCovariancePointer, MatrixStructure_Natural, lengthObservation, lengthObservation,
                               WeightCovariance);
    Math::Add(observationCovariancePointer, measureNoisePointer, lengthObservation * lengthObservation);

    // Pxy is transposed for later use
    Math::MatrixMultiplication(1.0,
                               sigmaPointsState.pointer[0u].GetPointer(), MatrixStructure_Natural, lengthState, sigmaPointsLength,
                               sigmaPointsObservation.pointer[0u].GetPointer(), MatrixStructure_Transposed, lengthObservation, sigmaPointsLength,
                               0.0, crossCovariancePointer, MatrixStructure_Transposed, lengthState, lengthObservation,
                               WeightCovariance);

    timer.Save();

    // K is transposed for later use. Solver to find K = Pxy*(Pyy^-1) <=> Pxy = K * Pyy <=> Pyy * K^T = Pxy^T => A*X=B
    std::cout << "Kalman Gain Calulation\n";
    Math::Solve(kalmanGainPointer, LinearSolverType_Cholesky, observationCovariancePointer, lengthObservation, lengthObservation, crossCovariancePointer, lengthObservation, lengthState);
    timer.Save();

    std::cout << "State Update\n";
    std::cout << "State\n";
    Math::Sub(measurePointer, observationPointer, lengthObservation);
    Math::MatrixMultiplication(1.0,
                               kalmanGainPointer, MatrixStructure_Transposed, lengthObservation, lengthState,
                               measurePointer, MatrixStructure_Natural, lengthObservation, 1u,
                               1.0, statePointer, MatrixStructure_Natural, lengthState, 1u);
    timer.Save();

    std::cout << "State Covariance\n";
    Math::MatrixMultiplication(1.0,
                               kalmanGainPointer, MatrixStructure_Transposed, lengthObservation, lengthState,
                               observationCovariancePointer, MatrixStructure_Natural, lengthObservation, lengthObservation,
                               0.0, crossCovariancePointer, MatrixStructure_Natural, lengthState, lengthObservation);

    Math::MatrixMultiplication(-1.0,
                               crossCovariancePointer, MatrixStructure_Natural, lengthState, lengthObservation,
                               kalmanGainPointer, MatrixStructure_Natural, lengthObservation, lengthState,
                               1.0, stateCovariancePointer, MatrixStructure_Natural, lengthState, lengthState);
    timer.Save();

    std::cout << "Delete Auxiliary Structures\n";
    MemoryHandler::Free(WeightMean);
    MemoryHandler::Free(WeightCovariance);

    MemoryHandler::Free(crossCovariancePointer);
    MemoryHandler::Free(kalmanGainPointer);

    MemoryHandler::Free(sigmaPointsState);
    MemoryHandler::Free(sigmaPointsObservation);

    MemoryHandler::Free(measure);
    MemoryHandler::Free(observation);
    MemoryHandler::Free(observationCovariance);
    timer.Save();

    std::cout << "Iteration Ended\n\n";
}