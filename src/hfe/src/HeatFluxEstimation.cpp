#include "../include/HeatFluxEstimation.hpp"

Pointer<HeatFluxEstimationMemory> HeatFluxEstimation::GetMemory()
{
    return memory;
}

HeatFluxEstimation::HeatFluxEstimation(
    unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt,
    double Sx, double Sy, double Sz, double St, double T0, double sT0, double sTm0, double Q0, double sQ0, PointerType type_in, PointerContext context_in)
{
    std::cout << "Parameter Initialization\n";
    parameter = MemoryHandler::AllocValue<Parameter, unsigned>(4u, type_in, context_in);
    unsigned indexL, indexD, indexS, indexAmp;
    indexL = parameter.pointer[0u].Add("Length", 4u, sizeof(unsigned));
    indexD = parameter.pointer[0u].Add("Delta", 4u, sizeof(double));
    indexS = parameter.pointer[0u].Add("Size", 4u, sizeof(double));
    indexAmp = parameter.pointer[0u].Add("Amp", 1u, sizeof(double));
    parameter.pointer[0u].Initialize(type_in, context_in);
    Pointer<unsigned> L = MemoryHandler::Alloc<unsigned>(4u, PointerType::CPU, PointerContext::CPU_Only);
    L.pointer[0u] = Lx;
    L.pointer[1u] = Ly;
    L.pointer[2u] = Lz;
    L.pointer[3u] = Lt;
    Pointer<double> D = MemoryHandler::Alloc<double>(4u, PointerType::CPU, PointerContext::CPU_Only);
    D.pointer[0u] = Sx / Lx;
    D.pointer[1u] = Sy / Ly;
    D.pointer[2u] = Sz / Lz;
    D.pointer[3u] = St / Lt;
    Pointer<double> S = MemoryHandler::Alloc<double>(4u, PointerType::CPU, PointerContext::CPU_Only);
    S.pointer[0u] = Sx;
    S.pointer[1u] = Sy;
    S.pointer[2u] = Sz;
    S.pointer[3u] = St;
    Pointer<double> Amp = MemoryHandler::Alloc<double>(1u, PointerType::CPU, PointerContext::CPU_Only);
    Amp.pointer[0u] = 5e4;

    parameter.pointer[0u].LoadData(indexL, L, 4u);
    parameter.pointer[0u].LoadData(indexD, D, 4u);
    parameter.pointer[0u].LoadData(indexS, S, 4u);
    parameter.pointer[0u].LoadData(indexAmp, Amp, 1u);
    std::cout << "Input Initialization\n";
    input = MemoryHandler::AllocValue<Data, unsigned>(2u, type_in, context_in);
    unsigned indexT, indexQ;
    indexT = input.pointer[0u].Add("Temperature", Lx * Ly * Lz);
    indexQ = input.pointer[0u].Add("Heat Flux", Lx * Ly);
    input.pointer[0u].Initialize(type_in, context_in);
    Pointer<double> T = MemoryHandler::Alloc<double>(Lx * Ly * Lz, type_in, context_in);
    Pointer<double> sigmaT = MemoryHandler::Alloc<double>(Lx * Ly * Lz, type_in, context_in);
    Pointer<double> Q = MemoryHandler::Alloc<double>(Lx * Ly, type_in, context_in);
    Pointer<double> sigmaQ = MemoryHandler::Alloc<double>(Lx * Ly, type_in, context_in);
    MemoryHandler::Set<double>(T, T0, 0, Lx * Ly * Lz);
    MemoryHandler::Set<double>(sigmaT, sT0, 0, Lx * Ly * Lz);
    MemoryHandler::Set<double>(Q, Q0, 0, Lx * Ly);
    MemoryHandler::Set<double>(sigmaQ, sQ0, 0, Lx * Ly);
    input.pointer[0u].LoadData(indexT, T, Lx * Ly * Lz);
    input.pointer[0u].LoadData(indexQ, Q, Lx * Ly);
    inputCovariance = MemoryHandler::AllocValue<DataCovariance, Data>(input.pointer[0], type_in, context_in);
    inputNoise = MemoryHandler::AllocValue<DataCovariance, Data>(input.pointer[0], type_in, context_in);
    inputCovariance.pointer[0].LoadData(indexT, sigmaT, Lx * Ly * Lz, DataCovarianceMode::Compact);
    inputCovariance.pointer[0].LoadData(indexQ, sigmaQ, Lx * Ly, DataCovarianceMode::Compact);
    inputNoise.pointer[0].LoadData(indexT, sigmaT, Lx * Ly * Lz, DataCovarianceMode::Compact);
    inputNoise.pointer[0].LoadData(indexQ, sigmaQ, Lx * Ly, DataCovarianceMode::Compact);

    MemoryHandler::Free<double>(sigmaQ);
    MemoryHandler::Free<double>(Q);
    MemoryHandler::Free<double>(sigmaT);
    MemoryHandler::Free<double>(T);

    std::cout << "Measure Initialization\n";
    measure = MemoryHandler::AllocValue<Data, unsigned>(1u, PointerType::CPU, PointerContext::CPU_Only);
    unsigned indexT_meas;
    indexT_meas = measure.pointer[0u].Add("Temperature", Lx * Ly);
    measure.pointer[0u].Initialize(type_in, context_in);

    Pointer<double> T_meas = MemoryHandler::Alloc<double>(Lx * Ly, PointerType::CPU, PointerContext::CPU_Only);
    Pointer<double> sigmaT_meas = MemoryHandler::Alloc<double>(Lx * Ly, PointerType::CPU, PointerContext::CPU_Only);
    MemoryHandler::Set<double>(T_meas, T0, 0, Lx * Ly);
    MemoryHandler::Set<double>(sigmaT_meas, sTm0, 0, Lx * Ly);
    measure.pointer[0u].LoadData(indexT_meas, T_meas, Lx * Ly);
    measureNoise = MemoryHandler::AllocValue<DataCovariance, Data>(measure.pointer[0u], PointerType::CPU, PointerContext::CPU_Only);
    measureNoise.pointer[0u].LoadData(indexT_meas, sigmaT_meas, Lx * Ly, DataCovarianceMode::Compact);

    MemoryHandler::Free<double>(sigmaT_meas);
    MemoryHandler::Free<double>(T_meas);

    std::cout << "Memory Initialization\n";
    memory = MemoryHandler::Alloc<HeatFluxEstimationMemory>(1u, PointerType::CPU, PointerContext::CPU_Only);
    memory.pointer[0u] = HeatFluxEstimationMemory(input.pointer[0u], inputCovariance.pointer[0u], inputNoise.pointer[0u], measure.pointer[0u], measureNoise.pointer[0u], parameter.pointer[0u], type_in, context_in);

    std::cout << "End Initialization\n";
}

void HeatFluxEstimation::UpdateMeasure(Pointer<double> T_in, unsigned Lx, unsigned Ly)
{
    PointerType type_in = PointerType::CPU;
    PointerContext context_in = PointerContext::CPU_Only;
    Pointer<Data> measure_aux = MemoryHandler::AllocValue<Data, unsigned>(1u, type_in, context_in);
    unsigned indexT_meas;
    indexT_meas = measure_aux.pointer[0u].Add("Temperature", Lx * Ly);
    measure_aux.pointer[0u].Initialize(type_in, context_in);
    measure_aux.pointer[0u].LoadData(indexT_meas, T_in, Lx * Ly);
    memory.pointer[0u].UpdateMeasure(measure_aux.pointer[0u]);
    MemoryHandler::Free<Data>(measure_aux);
}

HeatFluxEstimation::~HeatFluxEstimation()
{
    MemoryHandler::Free(memory);
    MemoryHandler::Free(parameter);
    MemoryHandler::Free(input);
    MemoryHandler::Free(inputCovariance);
    MemoryHandler::Free(inputNoise);
    MemoryHandler::Free(measure);
    MemoryHandler::Free(measureNoise);
}
