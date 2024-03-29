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
    state = MemoryHandler::AllocValue<Data, Data>(inputData_in, PointerType::CPU, PointerContext::CPU_Only);
    stateCovariance = MemoryHandler::AllocValue<DataCovariance, DataCovariance>(inputDataCovariance_in, PointerType::CPU, PointerContext::CPU_Only);
    stateNoise = MemoryHandler::AllocValue<DataCovariance, DataCovariance>(inputDataNoise_in, PointerType::CPU, PointerContext::CPU_Only);
    measureData = MemoryHandler::AllocValue<Data, Data>(measureData_in, PointerType::CPU, PointerContext::CPU_Only);
    measureDataNoise = MemoryHandler::AllocValue<DataCovariance, DataCovariance>(measureDataNoise_in, PointerType::CPU, PointerContext::CPU_Only);

    parameter = MemoryHandler::AllocValue<Parameter, Parameter>(inputParameter_in, PointerType::CPU, PointerContext::CPU_Only);

    type = type_in;
    context = context_in;
}

UKFMemory::UKFMemory(const UKFMemory &memory_in)
{
    state = MemoryHandler::Alloc<Data>(1u, PointerType::CPU, PointerContext::CPU_Only);
    stateCovariance = MemoryHandler::Alloc<DataCovariance>(1u, PointerType::CPU, PointerContext::CPU_Only);
    stateNoise = MemoryHandler::Alloc<DataCovariance>(1u, PointerType::CPU, PointerContext::CPU_Only);
    measureData = MemoryHandler::Alloc<Data>(1u, PointerType::CPU, PointerContext::CPU_Only);
    measureDataNoise = MemoryHandler::Alloc<DataCovariance>(1u, PointerType::CPU, PointerContext::CPU_Only);

    parameter = MemoryHandler::Alloc<Parameter>(1u, PointerType::CPU, PointerContext::CPU_Only);

    MemoryHandler::Copy(state, memory_in.state, 1u);
    MemoryHandler::Copy(stateCovariance, memory_in.stateCovariance, 1u);
    MemoryHandler::Copy(stateNoise, memory_in.stateNoise, 1u);
    MemoryHandler::Copy(measureData, memory_in.measureData, 1u);
    MemoryHandler::Copy(measureDataNoise, memory_in.measureDataNoise, 1u);
    MemoryHandler::Copy(parameter, memory_in.parameter, 1u);

    type = memory_in.type;
    context = memory_in.context;
}

UKFMemory &UKFMemory::operator=(const UKFMemory &memory_in)
{
    state = MemoryHandler::Alloc<Data>(1u, PointerType::CPU, PointerContext::CPU_Only);
    stateCovariance = MemoryHandler::Alloc<DataCovariance>(1u, PointerType::CPU, PointerContext::CPU_Only);
    stateNoise = MemoryHandler::Alloc<DataCovariance>(1u, PointerType::CPU, PointerContext::CPU_Only);
    measureData = MemoryHandler::Alloc<Data>(1u, PointerType::CPU, PointerContext::CPU_Only);
    measureDataNoise = MemoryHandler::Alloc<DataCovariance>(1u, PointerType::CPU, PointerContext::CPU_Only);

    parameter = MemoryHandler::Alloc<Parameter>(1u, PointerType::CPU, PointerContext::CPU_Only);

    MemoryHandler::Copy(state, memory_in.state, 1u);
    MemoryHandler::Copy(stateCovariance, memory_in.stateCovariance, 1u);
    MemoryHandler::Copy(stateNoise, memory_in.stateNoise, 1u);
    MemoryHandler::Copy(measureData, memory_in.measureData, 1u);
    MemoryHandler::Copy(measureDataNoise, memory_in.measureDataNoise, 1u);
    MemoryHandler::Copy(parameter, memory_in.parameter, 1u);

    type = memory_in.type;
    context = memory_in.context;
    return *this;
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
        std::cout << "Error: Measured data update is not valid or do not match the old size\n";
    }
}

Pointer<DataCovariance> UKFMemory::GetMeasureNoise()
{
    return measureDataNoise;
}
