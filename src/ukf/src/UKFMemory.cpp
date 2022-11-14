#include "../include/UKFMemory.hpp"

UKFMemory::UKFMemory()
{
    state = Pointer<Data>();
    stateCovariance = Pointer<DataCovariance>();
    stateNoise = Pointer<DataCovariance>();
    measureData = Pointer<Data>();
    measureDataNoise = Pointer<DataCovariance>();

    parameter = Pointer<Parameter>();

    type = PointerType::CPU;
    context = PointerContext::CPU_Only;
}

UKFMemory::UKFMemory(Data &inputData_in, DataCovariance &inputDataCovariance_in, DataCovariance &inputDataNoise_in, Data &measureData_in, DataCovariance &measureDataNoise_in, Parameter &inputParameter_in, PointerType type_in, PointerContext context_in)
{
    state = MemoryHandler::AllocValue<Data, Data>(inputData_in, type_in, context_in);
    stateCovariance = MemoryHandler::AllocValue<DataCovariance, DataCovariance>(inputDataCovariance_in, type_in, context_in);
    stateNoise = MemoryHandler::AllocValue<DataCovariance, DataCovariance>(inputDataNoise_in, type_in, context_in);
    measureData = MemoryHandler::AllocValue<Data, Data>(measureData_in, type_in, context_in);
    measureDataNoise = MemoryHandler::AllocValue<DataCovariance, DataCovariance>(measureDataNoise_in, type_in, context_in);

    parameter = MemoryHandler::AllocValue<Parameter, Parameter>(inputParameter_in, type_in, context_in);

    type = type_in;
    context = context_in;
}

UKFMemory::~UKFMemory()
{
    MemoryHandler::Free(measureDataNoise);
    MemoryHandler::Free(measureData);

    MemoryHandler::Free(stateNoise);
    MemoryHandler::Free(stateCovariance);
    MemoryHandler::Free(state);

    MemoryHandler::Free(parameter);
}

Pointer<Parameter> UKFMemory::GetParameter()
{
    return parameter;
}

Pointer<Data> UKFMemory::GetState()
{
    return state;
}

Pointer<DataCovariance> UKFMemory::GetStateCovariance()
{
    return stateCovariance;
}

Pointer<DataCovariance> UKFMemory::GetStateNoise()
{
    return stateNoise;
}

Pointer<Data> UKFMemory::GetMeasure()
{
    return measureData;
}

void UKFMemory::UpdateMeasure(Data &measureData_in)
{
    bool isValid = measureData_in.GetValidation();
    unsigned length = measureData.pointer[0u].GetLength();
    if (isValid && measureData_in.GetLength() == length)
    {
        Pointer<double> pointer = measureData.pointer[0u].GetPointer();
        Pointer<double> pointer_aux = measureData_in.GetPointer();
        MemoryHandler::Copy(pointer, pointer_aux, length);
    }
    else
    {
        std::cout << "Error: Measured data update is not valid or do not match the old size";
    }
}

Pointer<DataCovariance> UKFMemory::GetMeasureNoise()
{
    return measureDataNoise;
}
