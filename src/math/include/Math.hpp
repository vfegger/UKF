#ifndef MATH_HEADER
#define MATH_HEADER

#include <iostream>
#include "../../structure/include/MemoryHandler.hpp"

#include "MathEnum.hpp"
#include "MathCPU.hpp"
#include "MathGPU.hpp"

namespace Math
{
    // Auxiliary Functions
    void PrintVector(Pointer<double> vector_in, unsigned length_in, unsigned streamIndex_in = 0u);
    void PrintMatrix(Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, unsigned streamIndex_in = 0u);

    // In-Placed Calculation

    // Vector Element-wise Addition
    void Add(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in, unsigned streamIndex_in = 0u);
    // Vector Element-wise Subtraction
    void Sub(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in, unsigned streamIndex_in = 0u);
    // Vector Constant Multiplication
    void Mul(Pointer<double> vector_inout, double value_in, unsigned length_in, unsigned streamIndex_in = 0u);
    // Vector Element-wise Multiplication
    void Mul(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in, unsigned streamIndex_in = 0u);

    // Out-Placed Calculation

    // Vector Element-wise Addition
    void Add(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned streamIndex_in = 0u);
    // Vector Element-wise Subtraction
    void Sub(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned streamIndex_in = 0u);
    // Vector Constant Multiplication
    void Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in, unsigned streamIndex_in = 0u);
    // Vector Element-wise Multiplication
    void Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned streamIndex_in = 0u);

    // Matrix Multiplication
    void MatrixMultiplication(double alpha,
                              Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                              Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
                              double beta,
                              Pointer<double> matrix_out, MatrixStructure matrixOutStructure_in, unsigned lengthOutX_in, unsigned lengthOutY_in,
                              Pointer<double> weight_in = Pointer<double>(), unsigned cublasIndex_in = 0u, unsigned streamIndex_in = 0u);

    // Operators

    // In-Place Operator Distribution
    void Operation(void (*operation_in)(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned streamIndex_in),
                   Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                   unsigned strideOut_in = 1u, unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                   unsigned offsetOut_in = 0u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u,
                   unsigned streamIndex_in = 0u);
    void Operation(void (*operation_in)(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in, unsigned streamIndex_in),
                   Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                   unsigned strideOut_in = 1u, unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                   unsigned offsetOut_in = 0u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u,
                   unsigned streamIndex_in = 0u);
    // Out-Place Operator Distribution
    void Operation(void (*operation_in)(Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in, unsigned streamIndex_in),
                   Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                   unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                   unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u,
                   unsigned streamIndex_in = 0u);
    void Operation(void (*operation_in)(Pointer<double> vector_inout, double valueRight_in, unsigned length_in, unsigned streamIndex_in),
                   Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                   unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                   unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u,
                   unsigned streamIndex_in = 0u);

    // Reducibles Operations
    void Mean(Pointer<double> vector_out, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> weight_in = Pointer<double>(), unsigned cublasIndex_in = 0u, unsigned streamIndex_in = 0u);

    bool Compare(Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in);

    // Linear System Solvers

    // Helper Functions

    // Wrapper Methods
    void Decomposition(Pointer<double> decomposition_out, DecompositionType decompositionType_in, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> pivot_out = Pointer<double>(), unsigned cusolverIndex_in = 0u, unsigned streamIndex_in = 0u);

    void Solve(Pointer<double> X_out, LinearSolverType solverType_in, MatrixOperationSide operationSide_in,
               Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
               Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in,
               unsigned cusolverIndex_in = 0u, unsigned streamIndex_in = 0u, unsigned cublasIndex_in = 0u);

};

#endif