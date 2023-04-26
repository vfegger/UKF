#include "../include/HeatFluxEstimationMemory.hpp"

HFE_CRCMemory::HFE_CRCMemory() : UKFMemory(), iteration(1u)
{
}

HFE_CRCMemory::HFE_CRCMemory(Data &a, DataCovariance &b, DataCovariance &c, Data &d, DataCovariance &e, Parameter &f, PointerType type_in, PointerContext context_in, unsigned iteration) : UKFMemory(a, b, c, d, e, f, type_in, context_in), iteration(iteration)
{
}

HFE_CRCMemory::HFE_CRCMemory(const HFE_CRCMemory &memory_in) : UKFMemory(memory_in), iteration(memory_in.iteration)
{
}

void HFE_CRCMemory::Evolution(Data &data_inout, Parameter &parameter_in, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in)
{
    HCRC::HCRCProblem problem;
    Pointer<unsigned> length = parameter_in.GetPointer<unsigned>(0u);
    unsigned Lr = length.pointer[0u];
    unsigned Lth = length.pointer[1u];
    unsigned Lz = length.pointer[2u];
    unsigned Lt = length.pointer[3u];
    problem.Lr = Lr;
    problem.Lth = Lth;
    problem.Lz = Lz;
    problem.Lt = Lt;
    Pointer<double> delta = parameter_in.GetPointer<double>(1u);
    problem.dr = delta.pointer[0u];
    problem.dth = delta.pointer[1u];
    problem.dz = delta.pointer[2u];
    problem.dt = delta.pointer[3u];
    Pointer<double> parms = parameter_in.GetPointer<double>(3u);
    problem.amp = parms.pointer[0u];
    problem.r0 = parms.pointer[1u];
    problem.h = parms.pointer[2u];
    problem.iteration = iteration;
    Pointer<double> pointer = data_inout.GetPointer();
    Pointer<double> T_inout = data_inout[0u];
    Pointer<double> Q_in = data_inout[1u];
    Pointer<double> Tamb_in = data_inout[2u];
    double *workspace = NULL;
    if (pointer.type == PointerType::CPU)
    {
        HCRC::CPU::AllocWorkspaceEuler(workspace, problem.Lr * problem.Lth * problem.Lz);
        HCRC::CPU::Euler(T_inout.pointer, T_inout.pointer, Q_in.pointer, Tamb_in.pointer, problem, workspace);
        HCRC::CPU::FreeWorkspaceEuler(workspace);
    }
    else if (pointer.type == PointerType::GPU)
    {
        HCRC::GPU::AllocWorkspaceRK4(workspace, problem.Lr * problem.Lth * problem.Lz, stream_in);
        HCRC::GPU::RK4(T_inout.pointer, T_inout.pointer, Q_in.pointer, Tamb_in.pointer, problem, workspace, cublasHandle_in, stream_in);
        HCRC::GPU::FreeWorkspaceRK4(workspace, stream_in);
    }
}

void ObservationSimulation(Data &data_in, Parameter &parameter_in, Data &data_out, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in)
{
    unsigned Lr = parameter_in.GetPointer<unsigned>(0u).pointer[0u];
    unsigned Lth = parameter_in.GetPointer<unsigned>(0u).pointer[1u];
    unsigned Lz = parameter_in.GetPointer<unsigned>(0u).pointer[2u];
    unsigned length = Lth * Lz;
    unsigned *i = new unsigned[length];
    unsigned *j = new unsigned[length];
    unsigned *k = new unsigned[length];

    for (unsigned kk = 0u; kk < Lz; kk++)
    {
        for (unsigned jj = 0u; jj < Lth; jj++)
        {
            i[kk * Lth + jj] = 0u;
            j[kk * Lth + jj] = jj;
            k[kk * Lth + jj] = kk;
        }
    }

    Pointer<double> pointer_in = data_in.GetPointer();
    Pointer<double> pointer_out = data_out.GetPointer();
    Pointer<double> T_in = data_in[0u];
    Pointer<double> T_out = data_out[0u];

    Pointer<double> in, out;
    for (unsigned ii = 0u; ii < length; ii++)
    {
        in = Pointer<double>(T_in.pointer + HCRC::Index3D(i[ii], j[ii], k[ii], Lr, Lth, Lz), T_in.type, T_in.context);
        out = Pointer<double>(T_out.pointer + ii, T_out.type, T_out.context);
        MemoryHandler::Copy(in, out, 1u, stream_in);
    }
    delete[] k;
    delete[] j;
    delete[] i;
}

void ObservationMeasure(Data &data_in, Parameter &parameter_in, Data &data_out, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in)
{
    // Observation should use the correct sensors and angle positions
    unsigned Lr = parameter_in.GetPointer<unsigned>(0u).pointer[0u];
    unsigned Lth = parameter_in.GetPointer<unsigned>(0u).pointer[1u];
    unsigned Lz = parameter_in.GetPointer<unsigned>(0u).pointer[2u];
    double Sr = parameter_in.GetPointer<double>(2u).pointer[0u];
    double Sth = parameter_in.GetPointer<double>(2u).pointer[1u];
    double Sz = parameter_in.GetPointer<double>(2u).pointer[2u];
    double r0 = parameter_in.GetPointer<double>(3u).pointer[1u];
    double pi = acos(-1.0);

    // As the sensors are external of the object, the sensors are internal or external
    double r_int = parameter_in.GetPointer<double>(3u).pointer[1u];
    double r_ext = r_int + parameter_in.GetPointer<double>(1u).pointer[0u];
    // Angle is fixed as the sensor is spatially locked
    double th_1 = pi / 2.0;
    double th_2 = pi;
    // The sensors are in a given fixed position on the z axis
    double z_1 = 0.5 * Sz;
    double z_2 = 0.66 * Sz;
    double z_3 = 0.87 * Sz;
    // Project the positions on the grid
    unsigned length = HCRC_Measures;
    unsigned *i = new unsigned[length];
    unsigned *j = new unsigned[length];
    unsigned *k = new unsigned[length];
    unsigned it = 0u;
    // Sensor - 1 - (r_int,th_1,z_1)
    i[it] = std::max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = std::max((unsigned)(Lth * th_1 / Sth), Lth - 1u);
    k[it] = std::max((unsigned)(Lz * z_1 / Sz), Lz - 1u);
    it++;
    // Sensor - 2 - (r_int,th_1,z_2)
    i[it] = std::max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = std::max((unsigned)(Lth * th_1 / Sth), Lth - 1u);
    k[it] = std::max((unsigned)(Lz * z_2 / Sz), Lz - 1u);
    it++;
    // Sensor - 3 - (r_int,th_2,z_1)
    i[it] = std::max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = std::max((unsigned)(Lth * th_2 / Sth), Lth - 1u);
    k[it] = std::max((unsigned)(Lz * z_1 / Sz), Lz - 1u);
    it++;
    // Sensor - 4 - (r_int,th_1,z_3)
    i[it] = std::max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = std::max((unsigned)(Lth * th_1 / Sth), Lth - 1u);
    k[it] = std::max((unsigned)(Lz * z_3 / Sz), Lz - 1u);
    it++;

    Pointer<double> pointer_in = data_in.GetPointer();
    Pointer<double> pointer_out = data_out.GetPointer();
    Pointer<double> T_in = data_in[0u];
    Pointer<double> T_amb_in = data_in[2u];
    Pointer<double> T_out = data_out[0u];
    Pointer<double> T_amb_out = Pointer<double>(T_out.pointer + it, T_out.type, T_out.context);

    Pointer<double> in, out;
    for (unsigned ii = 0u; ii < it; ii++)
    {
        in = Pointer<double>(T_in.pointer + HCRC::Index3D(i[ii], j[ii], k[ii], Lr, Lth, Lz), T_in.type, T_in.context);
        out = Pointer<double>(T_out.pointer + ii, T_out.type, T_out.context);
        MemoryHandler::Copy(in, out, 1u, stream_in);
    }
    MemoryHandler::Copy(T_amb_out, T_amb_in, 1u, stream_in);

    delete[] k;
    delete[] j;
    delete[] i;
}

void HFE_CRCMemory::Observation(Data &data_in, Parameter &parameter_in, Data &data_out, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in)
{
    unsigned caseType = parameter_in.GetPointer<unsigned>(4u).pointer[0u];
    if (caseType == 0u)
    {
        ObservationMeasure(data_in, parameter_in, data_out, cublasHandle_in, cusolverHandle_in, stream_in);
    }
    else if (caseType == 1)
    {
        ObservationSimulation(data_in, parameter_in, data_out, cublasHandle_in, cusolverHandle_in, stream_in);
    }
}