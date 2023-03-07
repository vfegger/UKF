#ifndef HEAT_CONDUCTION_RADIATION_HEADER
#define HEAT_CONDUCTION_RADIATION_HEADER

#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <random>
#include "../../math/include/Math.hpp"

#define HCRC_Measures 10

namespace HCRC
{

    class HCRCProblem
    {
    public:
        double T0;
        double Q0;
        double amp;
        double r0;
        double Sr, Sth, Sz, St;
        double dr, dth, dz, dt;
        unsigned Lr, Lth, Lz, Lt;

        HCRCProblem();
        HCRCProblem(double T0_in, double Q0_in, double amp_in, double r0_in, double Sr_in, double Sth_in, double Sz_in, double St_in, unsigned Lr_in, unsigned Lth_in, unsigned Lz_in, unsigned Lt_in);
    };

    namespace CPU
    {
        double C(double T_in);
        double K(double T_in);
        double E(double T_in);
        double DifferentialK(const double T0_in, const double TP_in, const double delta_in, const double coef_in);
        void Differential(double *diff_out, const double *T_in, const double *Q_in, double amp, double r0, double dr, double dth, double dz, unsigned Lr, unsigned Lth, unsigned Lz);
        void AllocWorkspaceEuler(double *&workspace_out, unsigned length_in);
        void FreeWorkspaceEuler(double *&workspace_out);
        void Euler(double *T_out, const double *T_in, const double *Q_in, HCRCProblem &problem_in, double *workspace);
        void AllocWorkspaceRK4(double *&workspace_out, unsigned length_in);
        void FreeWorkspaceRK4(double *&workspace_out);
        void RK4(double *T_out, const double *T_in, const double *Q_in, HCRCProblem &problem_in, double *workspace);

        void SetFlux(double *Q_out, HCRCProblem &problem_in, unsigned t_in);
        void AddError(double *T_out, double mean_in, double sigma_in, unsigned length);

        void SelectTemperatures(double *T_out, double *T_in, unsigned *indexR_in, unsigned *indexTh_in, unsigned *indexZ_in, unsigned length_in, unsigned Lr, unsigned Lth, unsigned Lz);
    }

    namespace GPU
    {
        void Differential(double *diff_out, const double *T_in, const double *Q_in, double amp, double r0, double dr, double dth, double dz, unsigned Lr, unsigned Lth, unsigned Lz, cublasHandle_t handle_in, cudaStream_t stream_in = cudaStreamDefault);
        void AllocWorkspaceEuler(double *&workspace_out, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);
        void FreeWorkspaceEuler(double *&workspace_out, cudaStream_t stream_in = cudaStreamDefault);
        void Euler(double *T_out, double *T_in, double *Q_in, HCRCProblem &problem_in, double *workspace, cublasHandle_t handle_in, cudaStream_t stream_in = cudaStreamDefault);
        void AllocWorkspaceRK4(double *&workspace_out, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);
        void FreeWorkspaceRK4(double *&workspace_out, cudaStream_t stream_in = cudaStreamDefault);
        void RK4(double *T_out, double *T_in, double *Q_in, HCRCProblem &problem_in, double *workspace, cublasHandle_t handle_in, cudaStream_t stream_in = cudaStreamDefault);

        void SetFlux(double *Q_out, HCRCProblem &problem_in, unsigned t_in, cudaStream_t stream_in = cudaStreamDefault);
        void AddError(double *T_out, double mean_in, double sigma_in, unsigned length, cudaStream_t stream_in = cudaStreamDefault);

        void SelectTemperatures(double *T_out, double *T_in, unsigned *indexR_in, unsigned *indexTh_in, unsigned *indexZ_in, unsigned length_in, unsigned Lr, unsigned Lth, unsigned Lz);
    }
}

#endif