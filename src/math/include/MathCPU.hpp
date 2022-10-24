#ifndef MATHCPU_HEADER
#define MATHCPU_HEADER

#include <iostream>
#include "../include/MathEnum.hpp"
#include "../../structure/include/MemoryHandler.hpp"

namespace MathCPU
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
    void MatrixMultiplication(Pointer<double> matrix_out, double alpha, double beta,
                              Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                              Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
                              Pointer<double> weight_in = Pointer<double>());

    // Reducibles Operations
    void Mean(Pointer<double> vector_out, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> weight_in = Pointer<double>());

    bool Compare(Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in);

    // Linear System Solvers

    // Helper Functions

    // L*X=B (Matrices can be transposed by index without need of extra memory)
    void ForwardSubstitution(Pointer<double> matrix_out, MatrixStructure matrixStructure_in, unsigned lengthX_in, unsigned lengthY_in,
                             Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                             Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in);

    // U*X=B (Matrices can be transposed by index without need of extra memory)
    void BackwardSubstitution(Pointer<double> matrix_out, MatrixStructure matrixStructure_in, unsigned lengthX_in, unsigned lengthY_in,
                              Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                              Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in);

    // Main Methods

    void CholeskyDecomposition(Pointer<double> decomposition_out, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in);

    void CholeskySolver(Pointer<double> X_out, MatrixOperationSide operationSide_in,
                        Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
                        Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in);

    void GSSolver();

    // Wrapper Methods
    void Decomposition(Pointer<double> decomposition_out, DecompositionType decompositionType_in, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> pivot_out = Pointer<double>());

    void Solve(Pointer<double> X_out, LinearSolverType solverType_in, MatrixOperationSide operationSide_in,
               Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
               Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in);

};

#endif