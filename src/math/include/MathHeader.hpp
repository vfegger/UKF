#include <iostream>
#include "../../structure/include/MemoryHandler.hpp"

namespace Math {
    //In-Placed Calculation

    //Vector Element-wise Addition
    void Add();
    //Vector Element-wise Subtraction
    void Sub();
    //Vector Constant Multiplication
    void Mul();
    //Vector Element-wise Multiplication
    void Mul();

    //Out-Placed Calculation

    //Vector Element-wise Addition
    void Add();
    //Vector Element-wise Subtraction
    void Sub();
    //Vector Constant Multiplication
    void Mul();
    //Vector Element-wise Multiplication
    void Mul();

    //Matrix Multiplication
    void MatrixMultiplication();

    //Operators

    //In-Place Operator Distribution
    void Operation(void (*operation_in)(double *matrix_out, double *matrixLeft_in, double *matrixRight_in, unsigned length_in),
                   double *matrix_out, double *matrixLeft_in, double *matrixRight_in, unsigned length_in,
                   unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in = 1u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);
    //Out-Place Operator Distribution
    void Operation(void (*operation_in)(double *matrixLeft_inout, double *matrixRight_in, unsigned length_in),
                   double *matrixLeft_inout, double *matrixRight_in, unsigned length_in,
                   unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in = 1u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u);

    //Reducibles Operations
    void Mean();

    //Linear System Solvers

    //Helper Functions


    //Direct Methods
    void LU();
    void LUP();
    void QR();

    //Iterative Methods
    void GJ();
    void GS();

    //Wrapper Methods

}