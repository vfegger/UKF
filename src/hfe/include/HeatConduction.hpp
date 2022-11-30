#ifndef HEAT_CONDUCTION_HEADER
#define HEAT_CONDUCTION_HEADER

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <random>
#include "../../math/include/Math.hpp"

namespace HeatConduction
{

    class HeatConductionProblem
    {
    public:
        double T0;
        double Q0;
        double amp;
        double Sx, Sy, Sz, St;
        double dx, dy, dz, dt;
        unsigned Lx, Ly, Lz, Lt;

        HeatConductionProblem();
        HeatConductionProblem(double T0_in, double Q0_in, double amp_in, double Sx_in, double Sy_in, double Sz_in, double St_in, unsigned Lx_in, unsigned Ly_in, unsigned Lz_in, unsigned Lt_in);
    };

    namespace CPU
    {
        double C(double T_in);
        double K(double T_in);
        double DifferentialK(double TN_in, double T_in, double TP_in, double delta_in);
        void Differential(double *diff_out, const double *T_in, const double *Q_in, double amp, double dx, double dy, double dz, unsigned Lx, unsigned Ly, unsigned Lz);
        void AllocWorkspaceEuler(double *&workspace_out, unsigned length_in);
        void FreeWorkspaceEuler(double *&workspace_out);
        void Euler(double *T_out, const double *T_in, const double *Q_in, HeatConductionProblem &problem_in, double *workspace);
        void AllocWorkspaceRK4(double *&workspace_out, unsigned length_in);
        void FreeWorkspaceRK4(double *&workspace_out);
        void RK4(double *T_out, const double *T_in, const double *Q_in, HeatConductionProblem &problem_in, double *workspace);

        void SetFlux(double *Q_out, HeatConductionProblem &problem_in, unsigned t_in);
        void AddError(double *T_out, double mean_in, double sigma_in, unsigned length);
    }

    namespace GPU
    {
        static cudaStream_t stream;
        static cublasHandle_t handle;

        void Differential(double *diff_out, const double *T_in, const double *Q_in, double amp, double dx, double dy, double dz, unsigned Lx, unsigned Ly, unsigned Lz);
        void AllocWorkspaceEuler(double *&workspace_out, unsigned length_in);
        void FreeWorkspaceEuler(double *&workspace_out);
        void Euler(double *T_out, double *T_in, double *Q_in, HeatConductionProblem &problem_in, double *workspace);
        void AllocWorkspaceRK4(double *&workspace_out, unsigned length_in);
        void FreeWorkspaceRK4(double *&workspace_out);
        void RK4(double *T_out, double *T_in, double *Q_in, HeatConductionProblem &problem_in, double *workspace);

        void AllocWorkspaceEuler(double *&workspace_out, unsigned length_in);
        void FreeWorkspaceEuler(double *&workspace_out);
        void Euler(double *T_out, double *T_in, double *Q_in, HeatConductionProblem &problem_in, double *workspace);
        void AllocWorkspaceRK4(double *&workspace_out, unsigned length_in);
        void FreeWorkspaceRK4(double *&workspace_out);
        void RK4(double *T_out, double *T_in, double *Q_in, HeatConductionProblem &problem_in, double *workspace);

        void SetFlux(double *Q_out, HeatConductionProblem &problem_in, unsigned t_in);
        void AddError(double *T_out, double mean_in, double sigma_in, unsigned length);
    }
}

#endif