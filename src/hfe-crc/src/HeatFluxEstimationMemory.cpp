#include "../include/HeatFluxEstimationMemory.hpp"

HFE_CRCMemory::HFE_CRCMemory() : UKFMemory()
{
}

HFE_CRCMemory::HFE_CRCMemory(Data &a, DataCovariance &b, DataCovariance &c, Data &d, DataCovariance &e, Parameter &f, PointerType type_in, PointerContext context_in) : UKFMemory(a, b, c, d, e, f, type_in, context_in)
{
}

HFE_CRCMemory::HFE_CRCMemory(const HFE_CRCMemory &memory_in) : UKFMemory(memory_in)
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
    Pointer<double> pointer = data_inout.GetPointer();
    Pointer<double> T_inout = data_inout[0u];
    Pointer<double> Q_in = data_inout[1u];
    double *workspace = NULL;
    if (pointer.type == PointerType::CPU)
    {
        HCRC::CPU::AllocWorkspaceEuler(workspace, problem.Lr * problem.Lth * problem.Lz);
        HCRC::CPU::Euler(T_inout.pointer, T_inout.pointer, Q_in.pointer, problem, workspace);
        HCRC::CPU::FreeWorkspaceEuler(workspace);
    }
    else if (pointer.type == PointerType::GPU)
    {
        HCRC::GPU::AllocWorkspaceRK4(workspace, problem.Lr * problem.Lth * problem.Lz, stream_in);
        HCRC::GPU::RK4(T_inout.pointer, T_inout.pointer, Q_in.pointer, problem, workspace, cublasHandle_in, stream_in);
        HCRC::GPU::FreeWorkspaceRK4(workspace, stream_in);
    }
}

void HFE_CRCMemory::Observation(Data &data_in, Parameter &parameter_in, Data &data_out, cublasHandle_t cublasHandle_in, cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in)
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
    double th_1 = 3.0 * pi / 2.0;
    double th_2 = pi;
    // The sensors are in a given fixed position on the z axis
    double z_1 = 0.0 * Sz;
    double z_2 = 0.5 * Sz;
    double z_3 = 1.0 * Sz;
    // Project the positions on the grid
    unsigned length = 6u;
    unsigned *i = new unsigned[length];
    unsigned *j = new unsigned[length];
    unsigned *k = new unsigned[length];
    unsigned it = 0u;
    // Sensor - 1 - (r_int,th_1,z_1)
    i[it] = max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = max((unsigned)(Lth * th_1 / Sth), Lth - 1u);
    k[it] = max((unsigned)(Lz * z_1 / Sz), Lz - 1u);
    it++;
    // Sensor - 2 - (r_int,th_1,z_2)
    i[it] = max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = max((unsigned)(Lth * th_1 / Sth), Lth - 1u);
    k[it] = max((unsigned)(Lz * z_2 / Sz), Lz - 1u);
    it++;
    // Sensor - 3 - (r_int,th_1,z_3)
    i[it] = max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = max((unsigned)(Lth * th_1 / Sth), Lth - 1u);
    k[it] = max((unsigned)(Lz * z_3 / Sz), Lz - 1u);
    it++;
    // Sensor - 4 - (r_int,th_2,z_1)
    i[it] = max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = max((unsigned)(Lth * th_2 / Sth), Lth - 1u);
    k[it] = max((unsigned)(Lz * z_1 / Sz), Lz - 1u);
    it++;
    // Sensor - 5 - (r_int,th_2,z_2)
    i[it] = max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = max((unsigned)(Lth * th_2 / Sth), Lth - 1u);
    k[it] = max((unsigned)(Lz * z_2 / Sz), Lz - 1u);
    it++;
    // Sensor - 6 - (r_int,th_2,z_3)
    i[it] = max((unsigned)(Lr * (r_int - r0) / Sr), Lr - 1u);
    j[it] = max((unsigned)(Lth * th_2 / Sth), Lth - 1u);
    k[it] = max((unsigned)(Lz * z_3 / Sz), Lz - 1u);
    it++;

    Pointer<double> pointer = data_inout.GetPointer();
    Pointer<double> T_in = data_in[0u];
    Pointer<double> T_out = data_out[0u];
    if (pointer.type == PointerType::CPU)
    {
        HCRC::CPU::SelectTemperatures(T_out.pointer,T_in.pointer,i,j,k,it,Lr,Lth,Lz);
    }
    else if (pointer.type == PointerType::GPU)
    {
        HCRC::GPU::SelectTemperatures(T_out.pointer,T_in.pointer,i,j,k,it,Lr,Lth,Lz);
    }
}