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
    void Print(Pointer<double> vector_in, unsigned length_in);
    void Print(Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in);

    // In-Placed Calculation

    // Vector Element-wise Addition
    void Add(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in);
    // Vector Element-wise Subtraction
    void Sub(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in);
    // Vector Constant Multiplication
    void Mul(Pointer<double> vector_inout, double value_in, unsigned length_in);
    // Vector Element-wise Multiplication
    void Mul(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in);

    // Out-Placed Calculation

    // Vector Element-wise Addition
    void Add(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in);
    // Vector Element-wise Subtraction
    void Sub(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in);
    // Vector Constant Multiplication
    void Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in);
    // Vector Element-wise Multiplication
    void Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in);

    // Matrix Multiplication TODO
    void MatrixMultiplication(double alpha,
                              Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                              Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
                              double beta,
                              Pointer<double> matrix_out, MatrixStructure matrixOutStructure_in, unsigned lengthOutX_in, unsigned lengthOutY_in,
                              Pointer<double> weight_in = Pointer<double>());

    // Operators

    // In-Place Operator Distribution
    void Operation(void (*operation_in)(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in),
                   Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                   unsigned strideOut_in = 1u, unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                   unsigned offsetOut_in = 0u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);
    void Operation(void (*operation_in)(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in),
                   Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                   unsigned strideOut_in = 1u, unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                   unsigned offsetOut_in = 0u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);
    // Out-Place Operator Distribution
    void Operation(void (*operation_in)(Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in),
                   Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                   unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                   unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);
    void Operation(void (*operation_in)(Pointer<double> vector_inout, double valueRight_in, unsigned length_in),
                   Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                   unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                   unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);

    // Reducibles Operations
    void Mean(Pointer<double> vector_out, Pointer<double> matrixLeft_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> weight_in = Pointer<double>());

    bool Compare(Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in);

    // Linear System Solvers

    // Helper Functions

    // Main Methods

    // Wrapper Methods
    void Decomposition(Pointer<double> decomposition_out, DecompositionType decompositionType_in, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> pivot_out = Pointer<double>());

    void Solve(Pointer<double> X_out, LinearSolverType solverType_in, MatrixOperationSide operationSide_in,
               Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
               Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in);

};

#endif