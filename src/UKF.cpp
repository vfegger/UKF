#include "UKF.hpp"
#include <new>
#include "MathCustom.hpp"

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
    double* chol = new double[stateLength*stateLength]; 
    for(unsigned i = 0; i < stateLength * stateLength; i++){
        chol[i] = 0;
    }
    CholeskyDecomposition(covariance->GetStateCovariance(), chol, stateLength, stateLength);
    //TODO: Optimize
    for(unsigned i = 0u; i < sigmaLength; i++){
        sigmaPoints[i] = Point(mean);
    }
    for(unsigned i = 0u; i < stateLength; i++){
        double* aux = sigmaPoints[i+1u].GetState();
        for(unsigned j = 0u; j < stateLength; j++){
            aux[j] += chol[i*stateLength+j];  
        }
    }
    for(unsigned i = 0u; i < stateLength; i++){
        double* aux = sigmaPoints[i+stateLength+1u].GetState();
        for(unsigned j = 0u; j < stateLength; j++){
            aux[j] -= chol[i*stateLength+j];  
        }
    }
    delete[] chol;
    return;
}

void UKF::Solve(){
    Parameters* parameters = input->GetParameters();
    State* state = input->GetState();
    Measure* measure = input->GetMeasure();
    double* crossCovariance = NULL;
    unsigned lengthState = state->GetStateLength();
    unsigned lengthObservation = measure->GetStateLength();
    unsigned sigmaLength = 0u;
    unsigned auxMemorySize = (lengthState > lengthObservation) ? lengthState: lengthObservation;
    double* auxMemory = new double[auxMemorySize*auxMemorySize];

    // Generate Sigma Points of the state
    Point* sigmaPoints;
    Point* sigmaPointsObservation;
    SigmaPointsGenerator(state,sigmaPoints,sigmaLength);
    // Evolve all Sigma Points of the state
    for(unsigned i = 0; i < sigmaLength; i++){
        sigmaPoints[i].UpdateDataFromArray();
        input->Evolution(sigmaPoints[i].GetData(),parameters);
        sigmaPoints[i].UpdateArrayFromData();
    }
    // Get Mean of sigma points of the state
    double* meanState = state->GetPoint()->GetState();
    for(unsigned i = 0; i < lengthState; i++){
        meanState[i] = 0;
    }
    for(unsigned k = 0; k < sigmaLength; k++){
        double* aux = sigmaPoints[k].GetState();
        for(unsigned i = 0; i < lengthState; i++){
            meanState[i] += aux[i];
        }
    }
    // Calculate the covariance of the state
    double* covarianceState = state->GetPointCovariance()->GetStateCovariance();
    for(unsigned i = 0u; i < lengthState; i++){
        for(unsigned j = 0u; j < lengthState; j++){
            covarianceState[i*lengthState+j] = 0.0;
        }
    }
    for(unsigned k = 0; k < sigmaLength; k++){
        double* aux = sigmaPoints[k].GetState();
        for(unsigned i = 0u; i < lengthState; i++){
            for(unsigned j = 0; j < lengthState; j++){
                covarianceState[i*lengthState+j] += (aux[i]-meanState[i])*(aux[j]-meanState[j]);
            }
        }
    }
    // Point allocation
    sigmaPointsObservation = new(std::nothrow) Point[sigmaLength];
    for(unsigned i = 0u; i < sigmaLength; i++){
        sigmaPointsObservation[i] = Point(measure->GetRealPoint());
    }
    // Observe all Sigma Points of the state
    for(unsigned i = 0; i < sigmaLength; i++){
        sigmaPoints[i].UpdateDataFromArray();
        input->Observation(sigmaPoints[i].GetData(),parameters,sigmaPointsObservation[i].GetData());
        sigmaPointsObservation[i].UpdateArrayFromData();
    }
    // Get Mean of observation of all sigma Points
    double* meanObservation = measure->GetPoint()->GetState();
    double* realObservation = measure->GetRealPoint()->GetState();
    for(unsigned i = 0; i < lengthObservation; i++){
        meanObservation[i] = 0;
    }
    for(unsigned k = 0; k < sigmaLength; k++){
        double* aux = sigmaPointsObservation[k].GetState();
        for(unsigned i = 0; i < lengthObservation; i++){
            meanObservation[i] += aux[i];
        }
    }
    measure->GetRealPoint()->UpdateArrayFromData();
    // Calculate covariance of observation of the state
    double* covarianceObservation = measure->GetPointCovariance()->GetStateCovariance();
    for(unsigned i = 0u; i < lengthObservation; i++){
        for(unsigned j = 0u; j < lengthObservation; j++){
            covarianceObservation[i*lengthObservation+j] = 0.0;
        }
    }
    for(unsigned k = 0; k < sigmaLength; k++){
        double* aux = sigmaPoints[k].GetState();
        for(unsigned i = 0u; i < lengthObservation; i++){
            for(unsigned j = 0; j < lengthObservation; j++){
                covarianceObservation[i*lengthObservation+j] += (aux[i]-meanObservation[i])*(aux[j]-meanObservation[j]);
            }
        }
    }
    // Calculate cross-covariance between the state and the observation
    crossCovariance = new double[lengthState*lengthObservation];
    for(unsigned k = 0; k < sigmaLength; k++){
        double* auxState = sigmaPoints[k].GetState();
        double* auxObservation = sigmaPoints[k].GetState();
        for(unsigned i = 0u; i < lengthObservation; i++){
            for(unsigned j = 0; j < lengthState; j++){
                crossCovariance[i*lengthState+j] += (auxState[j]-meanState[j])*(auxObservation[i]-meanObservation[i]);
            }
        }
    }
    // Calculate the Kalman Gain
    // TODO: Optimize by right hand solver;
    double* kalmanGain = new double[lengthState*lengthObservation];
    Multiply(crossCovariance,
        PseudoInverse(covarianceObservation,auxMemory,lengthObservation,lengthObservation),
        kalmanGain,lengthState,lengthObservation,lengthObservation
    );

    // Update state and Covariance
    AddVector(meanState,
        Multiply(kalmanGain,
            SubVector(
                meanObservation,realObservation,lengthObservation),
            auxMemory,lengthState, 1u, lengthObservation),
        lengthState
    );
    AddVector(covarianceState,
        Multiply(kalmanGain,
            MultiplyTransposed(
                covarianceObservation,kalmanGain,auxMemory,lengthObservation,lengthState,lengthObservation),
            auxMemory, lengthState,lengthState, lengthObservation),
        lengthState*lengthState
    );

    delete[] kalmanGain;
    delete[] crossCovariance;
    delete[] sigmaPointsObservation;
    delete[] sigmaPoints;
    delete[] auxMemory;
    delete measure;
    delete state;
}

void UKF::Export(Output* output_out){
    output_out = output;
}