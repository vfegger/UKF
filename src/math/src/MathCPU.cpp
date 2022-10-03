#include "../include/MathCPU.hpp"

// In-Placed Calculation

// Vector Element-wise Addition
void MathCPU::Add(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        vector_inout.pointer[i] += vector_in.pointer[i];
    }
}
// Vector Element-wise Subtraction
void MathCPU::Sub(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        vector_inout.pointer[i] -= vector_in.pointer[i];
    }
}
// Vector Constant Multiplication
void MathCPU::Mul(Pointer<double> vector_inout, double value_in, unsigned length_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        vector_inout.pointer[i] *= value_in;
    }
}
// Vector Element-wise Multiplication
void MathCPU::Mul(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        vector_inout.pointer[i] *= vector_in.pointer[i];
    }
}

// Out-Placed Calculation

// Vector Element-wise Addition
void MathCPU::Add(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        vector_out.pointer[i] = vectorLeft_in.pointer[i] + vectorRight_in.pointer[i];
    }
}
// Vector Element-wise Subtraction
void MathCPU::Sub(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        vector_out.pointer[i] = vectorLeft_in.pointer[i] - vectorRight_in.pointer[i];
    }
}
// Vector Constant Multiplication
void MathCPU::Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        vector_out.pointer[i] = vectorLeft_in.pointer[i] * valueRight_in;
    }
}
// Vector Element-wise Multiplication
void MathCPU::Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        vector_out.pointer[i] = vectorLeft_in.pointer[i] * vectorRight_in.pointer[i];
    }
}
// Matrix Multiplication TODO
void GetIndexBoundaries(MatrixStructure matrixStructure_in, unsigned lengthX_in, unsigned lengthY_in, unsigned &sizeX_out, unsigned &sizeY_out, unsigned &strideX_out, unsigned &strideY_out)
{
    switch (matrixStructure_in)
    {
    case MatrixStructure::Natural:
        sizeX_out = lengthX_in;
        sizeY_out = lengthY_in;
        strideX_out = 1u;
        strideY_out = lengthX_in;
        break;
    case MatrixStructure::Transposed:
        sizeX_out = lengthY_in;
        sizeY_out = lengthX_in;
        strideX_out = lengthX_in;
        strideY_out = 1u;
        break;
    default:
        std::cout << "Error: Structure not defined for this operation.\n";
        return;
        break;
    }
}

void MathCPU::MatrixMultiplication(Pointer<double> matrix_out,
                                   Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                                   Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in)
{
    double *pA, *pB, *pC;
    unsigned M, N, KL, KR, K;
    unsigned sA1, sB1;
    unsigned sA2, sB2;
    pA = matrixLeft_in.pointer;
    pC = matrix_out.pointer;
    GetIndexBoundaries(matrixLeftStructure_in, lengthLeftX_in, lengthLeftY_in, M, KL, sA1, sA2);
    GetIndexBoundaries(matrixRightStructure_in, lengthRightX_in, lengthRightY_in, KR, N, sB1, sB2);
    if (KL == KR)
    {
        K = KL;
    }
    else
    {
        std::cout << "Error: Sizes do not match.\n";
        return;
    }
    pB = matrixRight_in.pointer;
    for (unsigned j = 0; j < N; j++)
    {
        pA = matrixLeft_in.pointer;
        for (unsigned i = 0; i < M; i++)
        {
            for (unsigned k = 0; k < K; k++)
            {
                pC[j * M + i] = *(pA + k * sA1) * *(pB + k * sB1);
            }
            pA += sA2;
        }
        pB += sB2;
    }
}

// Operators

// In-Place Operator Distribution
void MathCPU::Operation(void (*operation_in)(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in),
                        Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                        unsigned strideOut_in = 1u, unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                        unsigned offsetOut_in = 0u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u)
{
    vector_out.pointer += offsetOut_in;
    vectorLeft_in.pointer += offsetLeft_in;
    vectorRight_in.pointer += offsetRight_in;
    for (unsigned i = 0u; i < iteration_in; i++)
    {
        operation_in(vector_out, vectorLeft_in, vectorRight_in, length_in);
        vector_out.pointer += strideOut_in;
        vectorLeft_in.pointer += strideLeft_in;
        vectorRight_in.pointer += strideRight_in;
    }
}
void MathCPU::Operation(void (*operation_in)(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in),
                        Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                        unsigned strideOut_in = 1u, unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                        unsigned offsetOut_in = 0u, unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u)
{
    vector_out.pointer += offsetOut_in;
    vectorLeft_in.pointer += offsetLeft_in;
    vectorRight_in.pointer += offsetRight_in;
    for (unsigned i = 0u; i < iteration_in; i++)
    {
        operation_in(vector_out, vectorLeft_in, *vectorRight_in.pointer, length_in);
        vector_out.pointer += strideOut_in;
        vectorLeft_in.pointer += strideLeft_in;
        vectorRight_in.pointer += strideRight_in;
    }
}
// Out-Place Operator Distribution
void MathCPU::Operation(void (*operation_in)(Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in),
                        Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                        unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                        unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u)
{
    vector_inout.pointer += offsetLeft_in;
    vectorRight_in.pointer += offsetRight_in;
    for (unsigned i = 0u; i < iteration_in; i++)
    {
        operation_in(vector_inout, vectorRight_in, length_in);
        vector_inout.pointer += strideLeft_in;
        vectorRight_in.pointer += strideRight_in;
    }
}
void MathCPU::Operation(void (*operation_in)(Pointer<double> vector_inout, double valueRight_in, unsigned length_in),
                        Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                        unsigned strideLeft_in = 1u, unsigned strideRight_in = 1u,
                        unsigned offsetLeft_in = 0u, unsigned offsetRight_in = 0u)
{
    vector_inout.pointer += offsetLeft_in;
    vectorRight_in.pointer += offsetRight_in;
    for (unsigned i = 0u; i < iteration_in; i++)
    {
        operation_in(vector_inout, *vectorRight_in.pointer, length_in);
        vector_inout.pointer += strideLeft_in;
        vectorRight_in.pointer += strideRight_in;
    }
}

// Reducibles Operations
void MathCPU::Mean(Pointer<double> vector_out, Pointer<double> matrixLeft_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> weight_in = Pointer<double>())
{
    for (unsigned i = 0u; i < lengthX_in; i++)
    {
        vector_out.pointer[i] = 0.0;
    }
    if (weight_in.pointer == NULL)
    {
        for (unsigned j = 0u; j < lengthY_in; j++)
        {
            for (unsigned i = 0u; i < lengthX_in; i++)
            {
                vector_out.pointer[i] += matrixLeft_in.pointer[j * lengthX_in + i];
            }
        }
        for (unsigned i = 0u; i < lengthX_in; i++)
        {
            vector_out.pointer[i] *= (1.0/lengthY_in);
        }
    }
    else
    {
        for (unsigned j = 0u; j < lengthY_in; j++)
        {
            for (unsigned i = 0u; i < lengthX_in; i++)
            {
                vector_out.pointer[j * lengthX_in + i] = matrixLeft_in.pointer[j * lengthX_in + i] * weight_in.pointer[j];
            }
        }
    }
}

// Linear System Solvers

// Helper Functions

// Wrapper Methods
void MathCPU::Decomposition(Pointer<double> decomposition_out, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> pivot_out = Pointer<double>())
{
}

void MathCPU::Solve(Pointer<double> X_out, LinearSolverType solverType_in, MatrixOperationSide operationSide_in,
                    Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
                    Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in)
{
}
