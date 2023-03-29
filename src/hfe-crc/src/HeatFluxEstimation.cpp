#include "../include/HeatFluxEstimation.hpp"

Pointer<HFE_CRCMemory> HFE_CRC::GetMemory()
{
    return memory;
}

HFE_CRC::HFE_CRC(unsigned Lr, unsigned Lth, unsigned Lz, unsigned Lt, double Sr, double Sth, double Sz, double St, double T0, double sT0, double sTm0, double Q0, double sQ0, double Tamb0, double sTamb0, double Amp, double r0, double h, PointerType type_in, PointerContext context_in)
{
    std::cout << "Parameter Initialization\n";
    parameter = MemoryHandler::AllocValue<Parameter, unsigned>(4u, PointerType::CPU, PointerContext::CPU_Only);
    unsigned indexL, indexD, indexS, indexP;
    indexL = parameter.pointer[0u].Add("Length", 4u, sizeof(unsigned));
    indexD = parameter.pointer[0u].Add("Delta", 4u, sizeof(double));
    indexS = parameter.pointer[0u].Add("Size", 4u, sizeof(double));
    indexP = parameter.pointer[0u].Add("External Parms", 3u, sizeof(double));
    parameter.pointer[0u].Initialize(PointerType::CPU, PointerContext::CPU_Only);
    Pointer<unsigned> L = MemoryHandler::Alloc<unsigned>(4u, PointerType::CPU, PointerContext::CPU_Only);
    L.pointer[0u] = Lr;
    L.pointer[1u] = Lth;
    L.pointer[2u] = Lz;
    L.pointer[3u] = Lt;
    Pointer<double> D = MemoryHandler::Alloc<double>(4u, PointerType::CPU, PointerContext::CPU_Only);
    D.pointer[0u] = Sr / Lr;
    D.pointer[1u] = Sth / Lth;
    D.pointer[2u] = Sz / Lz;
    D.pointer[3u] = St / Lt;
    Pointer<double> S = MemoryHandler::Alloc<double>(4u, PointerType::CPU, PointerContext::CPU_Only);
    S.pointer[0u] = Sr;
    S.pointer[1u] = Sth;
    S.pointer[2u] = Sz;
    S.pointer[3u] = St;
    Pointer<double> P = MemoryHandler::Alloc<double>(3u, PointerType::CPU, PointerContext::CPU_Only);
    P.pointer[0u] = Amp; // Amp
    P.pointer[1u] = r0;  // Radius
    P.pointer[2u] = h;   // Heat Convection Coefficient

    parameter.pointer[0u].LoadData(indexL, L, 4u);
    parameter.pointer[0u].LoadData(indexD, D, 4u);
    parameter.pointer[0u].LoadData(indexS, S, 4u);
    parameter.pointer[0u].LoadData(indexP, P, 3u);

    MemoryHandler::Free<unsigned>(L);
    MemoryHandler::Free<double>(D);
    MemoryHandler::Free<double>(S);
    MemoryHandler::Free<double>(P);

    std::cout << "Input Initialization\n";
    input = MemoryHandler::AllocValue<Data, unsigned>(3u, PointerType::CPU, PointerContext::CPU_Only);
    unsigned indexT, indexQ, indexTamb;
    indexT = input.pointer[0u].Add("Temperature", Lr * Lth * Lz);
    indexQ = input.pointer[0u].Add("Heat Flux", Lth * Lz);
    indexTamb = input.pointer[0u].Add("Temperature Ambient", 1u);
    input.pointer[0u].Initialize(type_in, context_in);
    Pointer<double> T = MemoryHandler::Alloc<double>(Lr * Lth * Lz, type_in, context_in);
    Pointer<double> sigmaT = MemoryHandler::Alloc<double>(Lr * Lth * Lz, type_in, context_in);
    Pointer<double> Q = MemoryHandler::Alloc<double>(Lth * Lz, type_in, context_in);
    Pointer<double> sigmaQ = MemoryHandler::Alloc<double>(Lth * Lz, type_in, context_in);
    Pointer<double> Tamb = MemoryHandler::Alloc<double>(1u, type_in, context_in);
    Pointer<double> sigmaTamb = MemoryHandler::Alloc<double>(1u, type_in, context_in);
    MemoryHandler::Set<double>(T, T0, 0u, Lr * Lth * Lz);
    MemoryHandler::Set<double>(sigmaT, sT0, 0u, Lr * Lth * Lz);
    MemoryHandler::Set<double>(Q, Q0, 0u, Lth * Lz);
    MemoryHandler::Set<double>(sigmaQ, sQ0, 0u, Lth * Lz);
    MemoryHandler::Set<double>(Tamb, Tamb0, 0u, 1u);
    MemoryHandler::Set<double>(sigmaTamb, sTamb0, 0u, 1u);
    input.pointer[0u].LoadData(indexT, T, Lr * Lth * Lz);
    input.pointer[0u].LoadData(indexQ, Q, Lth * Lz);
    input.pointer[0u].LoadData(indexTamb, Tamb, 1u);
    inputCovariance = MemoryHandler::AllocValue<DataCovariance, Data>(input.pointer[0], PointerType::CPU, PointerContext::CPU_Only);
    inputNoise = MemoryHandler::AllocValue<DataCovariance, Data>(input.pointer[0], PointerType::CPU, PointerContext::CPU_Only);
    inputCovariance.pointer[0].LoadData(indexT, sigmaT, Lr * Lth * Lz, DataCovarianceMode::Compact);
    inputCovariance.pointer[0].LoadData(indexQ, sigmaQ, Lth * Lz, DataCovarianceMode::Compact);
    inputCovariance.pointer[0].LoadData(indexTamb, sigmaTamb, 1u, DataCovarianceMode::Compact);
    inputNoise.pointer[0].LoadData(indexT, sigmaT, Lr * Lth * Lz, DataCovarianceMode::Compact);
    inputNoise.pointer[0].LoadData(indexQ, sigmaQ, Lth * Lz, DataCovarianceMode::Compact);
    inputNoise.pointer[0].LoadData(indexTamb, sigmaTamb, 1u, DataCovarianceMode::Compact);

    MemoryHandler::Free<double>(sigmaTamb);
    MemoryHandler::Free<double>(Tamb);
    MemoryHandler::Free<double>(sigmaQ);
    MemoryHandler::Free<double>(Q);
    MemoryHandler::Free<double>(sigmaT);
    MemoryHandler::Free<double>(T);

    std::cout << "Measure Initialization\n";
    measure = MemoryHandler::AllocValue<Data, unsigned>(1u, PointerType::CPU, PointerContext::CPU_Only);
    unsigned indexT_meas;
    indexT_meas = measure.pointer[0u].Add("Temperature", HCRC_Measures);
    measure.pointer[0u].Initialize(type_in, context_in);

    Pointer<double> T_meas = MemoryHandler::Alloc<double>(HCRC_Measures, type_in, context_in);
    Pointer<double> sigmaT_meas = MemoryHandler::Alloc<double>(HCRC_Measures, type_in, context_in);
    MemoryHandler::Set<double>(T_meas, T0, 0, HCRC_Measures);
    MemoryHandler::Set<double>(sigmaT_meas, sTm0, 0, HCRC_Measures);
    measure.pointer[0u].LoadData(indexT_meas, T_meas, HCRC_Measures);
    measureNoise = MemoryHandler::AllocValue<DataCovariance, Data>(measure.pointer[0u], PointerType::CPU, PointerContext::CPU_Only);
    measureNoise.pointer[0u].LoadData(indexT_meas, sigmaT_meas, HCRC_Measures, DataCovarianceMode::Compact);

    MemoryHandler::Free<double>(sigmaT_meas);
    MemoryHandler::Free<double>(T_meas);

    std::cout << "Memory Initialization\n";
    memory = MemoryHandler::Alloc<HFE_CRCMemory>(1u, PointerType::CPU, PointerContext::CPU_Only);
    memory.pointer[0u] = HFE_CRCMemory(input.pointer[0u], inputCovariance.pointer[0u], inputNoise.pointer[0u], measure.pointer[0u], measureNoise.pointer[0u], parameter.pointer[0u], type_in, context_in);

    std::cout << "End Initialization\n";
}

void HFE_CRC::UpdateMeasure(Pointer<double> T_in, PointerType type_in, PointerContext context_in)
{
    Pointer<Data> measure_aux = MemoryHandler::AllocValue<Data, unsigned>(1u, PointerType::CPU, PointerContext::CPU_Only);
    unsigned indexT_meas;
    indexT_meas = measure_aux.pointer[0u].Add("Temperature", HCRC_Measures);
    measure_aux.pointer[0u].Initialize(type_in, context_in);
    measure_aux.pointer[0u].LoadData(indexT_meas, T_in, HCRC_Measures);
    memory.pointer[0u].UpdateMeasure(measure_aux.pointer[0u]);
    MemoryHandler::Free<Data>(measure_aux);
}

HFE_CRC::~HFE_CRC()
{
    MemoryHandler::Free(memory);
    MemoryHandler::Free(parameter);
    MemoryHandler::Free(input);
    MemoryHandler::Free(inputCovariance);
    MemoryHandler::Free(inputNoise);
    MemoryHandler::Free(measure);
    MemoryHandler::Free(measureNoise);
}
