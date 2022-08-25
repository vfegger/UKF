#ifndef MATH_GPU_HEADER
#define MATH_GPU_HEADER

#include "MathEnum.hpp"
#include <iostream>
#include <iomanip>

namespace MathGPU {
    void PrintMatrix(double* matrix_in, unsigned lengthX_in, unsigned lengthY_in);

    //Out-Placed Calculation
    void Add(double* vector_out, double* vectorLeft_in, double* vectorRight_in, unsigned length_in);
    void Sub(double* vector_out, double* vectorLeft_in, double* vectorRight_in, unsigned length_in);
    void Mul(double* vector_out, double* vectorLeft_in, double* vectorRight_in, unsigned length_in);
    void Mul(double* vector_out, double* vectorLeft_in, double value_in, unsigned length_in);

    //In-Placed Calculation
    void Add(double* vector_inout, double* vector_in, unsigned length_in);
    void Sub(double* vector_inout, double* vector_in, unsigned length_in);
    void Mul(double* vector_inout, double* vector_in, unsigned length_in);
    void Mul(double* vector_inout, double value_in, unsigned length_in);

    //Matrix Multiplication
    void MatrixMultiplication(double* matrix_out, double alpha, double beta,
    double* matrixLeft_in, MatrixStructure structureLeft, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, MatrixStructure structureRight, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in = NULL);
    
    //Operators
    void Operation(void (*operation_in)(double* matrix_out, double* matrixLeft_in, double* matrixRight_in, unsigned length_in),
    double* matrix_out, double* matrixLeft_in, double* matrixRight_in, unsigned length_in,
    unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in = 1u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);
    void Operation(void (*operation_in)(double* matrixLeft_inout, double* matrixRight_in, unsigned length_in),
    double* matrixLeft_inout, double* matrixRight_in, unsigned length_in,
    unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in = 1u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);

    //Reducibles Operations
    double Mean(double* vector_in, unsigned length_in, double* weight_in = NULL);
    void Mean(double* vector_out, double* matrix_in, unsigned lengthX_in, unsigned lengthY_in, double* weight_in = NULL);

    //Linear System Solvers
    void ForwardSubstitution(double* X_out, double* A, double* B,
    unsigned lengthAX_in, unsigned lengthAY_in,
    unsigned lengthBX_in, unsigned lengthBY_in,
    unsigned lengthXX_in, unsigned lengthXY_in);
    void BackwardSubstitution(double* X_out, double* A, double* B,
    unsigned lengthAX_in, unsigned lengthAY_in,
    unsigned lengthBX_in, unsigned lengthBY_in,
    unsigned lengthXX_in, unsigned lengthXY_in);

    //Direct Methods
    void LU();
    void LUP();
    void QR();
    
    //Iterative Methods
    void GJ();
    void GS();

    //Wrapper Methods
    void LHSolver(double* X_out, double* A, double* B,
    unsigned lengthAX_in, unsigned lengthAY_in,
    unsigned lengthBX_in, unsigned lengthBY_in,
    unsigned lengthXX_in, unsigned lengthXY_in,
    double tol = 0.0001);
    void RHSolver(double* X_out, double* A, double* B,
    unsigned lengthAX_in, unsigned lengthAY_in,
    unsigned lengthBX_in, unsigned lengthBY_in,
    unsigned lengthXX_in, unsigned lengthXY_in,
    double tol = 0.0001);
}

#endif