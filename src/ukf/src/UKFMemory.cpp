#include "../include/UKFMemory.hpp"

UKFMemory::UKFMemory(Data& inputData_in, Data& inputDataCovariance_in, Data& inputDataNoise_in, Data& measureData_in, Data& measureDataNoise_in, Parameter& inputParameter_in){
    state = new Data(inputData_in);
    stateCovariance = new Data(inputDataCovariance_in);
    stateNoise = new Data(inputDataNoise_in);
    measureData = new Data(measureData_in);
    measureDataNoise = new Data(measureDataNoise_in);

    parameter = new Parameter(inputParameter_in);
}

UKFMemory::~UKFMemory(){
    delete measureDataNoise;
    delete measureData;

    delete stateNoise;
    delete stateCovariance;
    delete state;
    
    delete parameter;
}

Parameter* UKFMemory::GetParameter(){
    return parameter;
}

Data* UKFMemory::GetState(){
    return state;
}

Data* UKFMemory::GetStateCovariance(){
    return stateCovariance;
}

Data* UKFMemory::GetStateNoise(){
    return stateCovariance;
}

Data* UKFMemory::GetMeasure(){
    return measureData;
}

void UKFMemory::UpdateMeasure(Data& measureData_in){
    bool isValid = measureData_in.GetValidation();
    unsigned length = measureData->GetLength();
    if(isValid && measureData_in.GetLength() == length){
        double* pointer = measureData->GetPointer();
        double* pointer_aux = measureData_in.GetPointer();
        for(unsigned i = 0u; i < length; i++){
            pointer[i] = pointer_aux[i];
        }
    } else {
        std::cout << "Error: Measured data update is not valid or do not match the old size";
    }
}

Data* UKFMemory::GetMeasureNoise(){
    return measureDataNoise;
}

