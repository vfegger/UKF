#include "../include/UKFMemory.hpp"

UKFMemory::UKFMemory(Data& inputData_in, DataCovariance& inputDataCovariance_in, DataCovariance& inputDataNoise_in, Data& measureData_in, DataCovariance& measureDataNoise_in, Parameter& inputParameter_in){
    state = new Data(inputData_in);
    stateCovariance = new DataCovariance(inputDataCovariance_in);
    stateNoise = new DataCovariance(inputDataNoise_in);
    measureData = new Data(measureData_in);
    measureDataNoise = new DataCovariance(measureDataNoise_in);

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

DataCovariance* UKFMemory::GetStateCovariance(){
    return stateCovariance;
}

DataCovariance* UKFMemory::GetStateNoise(){
    return stateNoise;
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

DataCovariance* UKFMemory::GetMeasureNoise(){
    return measureDataNoise;
}

