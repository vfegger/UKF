#ifndef MATH_HEADER
#define MATH_HEADER

#include "../../structure/include/Data.hpp"
#include <math.h>
#include <iomanip>
#include <omp.h>

namespace Math {
    enum MatrixStructure{
        Natural,Transposed,
    };

    void PrintMatrix(double* matrix_in, unsigned lengthX_in, unsigned lengthY_in);

    void CholeskyDecomposition(double* matrix_out, double* matrix_in, unsigned lengthX_in, unsigned lengthY_in);
    void AddInPlace(double* matrix_inout, double* matrix_in, unsigned length_in);
    void SubInPlace(double* matrix_inout, double* matrix_in, unsigned length_in);
    void ConstantMultiplicationInPlace(double* matrix_inout, double value_in, unsigned length_in);

    //Matrix Multiplication alpha * A * B + beta * C => C
    void MatrixMultiplication(double* matrix_out, double alpha, double beta,
    double* matrixLeft_in, MatrixStructure structureLeft, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, MatrixStructure structureRight, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in = NULL);
    void DistributeOperation(void (*f)(double* matrixLeft_inout, double* matrixRight_in, unsigned length_in), double* matrixLeft_inout, double* matrixRight_in, unsigned length_in, unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in = 1u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);
    void Mean(double* vector_out, double* matrix_in, unsigned lengthX_in, unsigned lengthY_in, double* weight_in = NULL);

    // Solves X*A = B for X
    void RHSolver(double* X_out, double* A, double* B, unsigned lengthX_in, unsigned lengthY_in, double tol = 0.0001);
}

#ifdef _OPENMP

namespace MathOpenMP{
    enum MatrixStructure{
        Natural,Transposed,
    };

    
    void PrintMatrix(double* matrix_in, unsigned lengthX_in, unsigned lengthY_in);

    void CholeskyDecomposition(double* matrix_out, double* matrix_in, unsigned lengthX_in, unsigned lengthY_in);
    void AddInPlace(double* matrix_inout, double* matrix_in, unsigned length_in);
    void SubInPlace(double* matrix_inout, double* matrix_in, unsigned length_in);
    void ConstantMultiplicationInPlace(double* matrix_inout, double value_in, unsigned length_in);
    
    //Matrix Multiplication alpha * A * B + beta * C => C
    void MatrixMultiplication(double* matrix_out, double alpha, double beta,
    double* matrixLeft_in, MatrixStructure structureLeft, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, MatrixStructure structureRight, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in = NULL);
    
    void DistributeOperation(void (*f)(double* matrixLeft_inout, double* matrixRight_in, unsigned length_in), double* matrixLeft_inout, double* matrixRight_in, unsigned length_in, unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in = 1u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);
    void Mean(double* vector_out, double* matrix_in, unsigned lengthX_in, unsigned lengthY_in, double* weight_in = NULL);

    // Solves X*A = B for X
    void RHSolver(double* X_out, double* A, double* B, unsigned lengthX_in, unsigned lengthY_in, double tol = 0.0001);

}

#endif

#endif