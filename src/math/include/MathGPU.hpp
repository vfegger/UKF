#ifndef MATHGPU_HEADER
#define MATHGPU_HEADER

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../include/MathEnum.hpp"
#include "../../structure/include/MemoryHandler.hpp"

namespace MathGPU
{
    // Auxiliary Functions
    void Print(Pointer<double> vector_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);
    void Print(Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, cudaStream_t stream_in = cudaStreamDefault);

    // In-Placed Calculation

    // Vector Element-wise Addition
    void Add(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);
    // Vector Element-wise Subtraction
    void Sub(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);
    // Vector Constant Multiplication
    void Mul(Pointer<double> vector_inout, double value_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);
    // Vector Element-wise Multiplication
    void Mul(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);

    // Out-Placed Calculation

    // Vector Element-wise Addition
    void Add(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);
    // Vector Element-wise Subtraction
    void Sub(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);
    // Vector Constant Multiplication
    void Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);
    // Vector Element-wise Multiplication
    void Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);

    // Matrix Multiplication
    void MatrixMultiplication(double alpha,
                              Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                              Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
                              double beta,
                              Pointer<double> matrix_out, MatrixStructure matrixOutStructure_in, unsigned lengthOutX_in, unsigned lengthOutY_in,
                              cublasHandle_t handle_in, cudaStream_t stream_in = cudaStreamDefault, Pointer<double> weight_in = Pointer<double>());

    // Operators

    // Reducibles Operations
    void Mean(Pointer<double> vector_out, Pointer<double> matrixLeft_in, unsigned lengthX_in, unsigned lengthY_in, cublasHandle_t handle_in, cudaStream_t stream_in = cudaStreamDefault, Pointer<double> weight_in = Pointer<double>());

    bool Compare(Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, cudaStream_t stream_in = cudaStreamDefault);

    // Linear System Solvers

    // Helper Functions

    // Main Methods

    // Wrapper Methods
    void Decomposition(Pointer<double> decomposition_out, DecompositionType decompositionType_in, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, cusolverDnHandle_t handle_in, cudaStream_t stream_in = cudaStreamDefault, Pointer<double> pivot_out = Pointer<double>());

    void Solve(Pointer<double> X_out, LinearSolverType solverType_in, MatrixOperationSide operationSide_in,
               Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
               Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in,
               cusolverDnHandle_t cusolverHandle_in, cudaStream_t stream_in, cublasHandle_t cublasHandle_in);

};

#endif