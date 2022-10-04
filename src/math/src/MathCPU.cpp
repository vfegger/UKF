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
                pC[j * M + i] = (*(pA + k * sA1)) * (*(pB + k * sB1));
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
            vector_out.pointer[i] *= (1.0 / lengthY_in);
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

void MathCPU::ForwardSubstitution(Pointer<double> matrix_out, MatrixStructure matrixStructure_in, unsigned lengthX_in, unsigned lengthY_in,
                                  Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                                  Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in)
{
    double *A, *X, *B;
    unsigned MA, KA, KX, NX, MB, NB, M, K, N;
    unsigned sA1, sA2, sX1, sX2, sB1, sB2;
    GetIndexBoundaries(matrixLeftStructure_in, lengthLeftX_in, lengthLeftY_in, MA, KA, sA1, sA2);
    GetIndexBoundaries(matrixStructure_in, lengthX_in, lengthY_in, KX, NX, sX1, sX2);
    GetIndexBoundaries(matrixRightStructure_in, lengthRightX_in, lengthRightY_in, MB, NB, sB1, sB2);

    if (MA != MB || KA != KX || NX != NB)
    {
        std::cout << "Error: sizes do not match.\n";
        return;
    }
    else if (MA != KA)
    {
        std::cout << "Error: matrix is not a square matrix.\n";
        return;
    }
    A = matrixLeft_in.pointer;
    X = matrix_out.pointer;
    B = matrixRight_in.pointer;
    M = MA;
    K = KA;
    N = NX;
    for (unsigned j = 0u; j < N; j++)
    {
        for (unsigned k = 0u; k < K; k++)
        {
            double sum = 0.0;
            for (unsigned i = 0u; i < k; i++)
            {
                sum += (*(A + i * sA2 + k * sA1)) * (*(X + j * sX2 + i * sX1));
            }
            *(X + j * sX2 + k * sX1) = (1.0 / (*(A + k * sA2 + k * sA1))) * (*(B + j * sB2 + k * sB1) - sum);
        }
    }
}

void MathCPU::BackwardSubstitution(Pointer<double> matrix_out, MatrixStructure matrixStructure_in, unsigned lengthX_in, unsigned lengthY_in,
                                   Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                                   Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in)
{
    double *A, *X, *B;
    unsigned MA, KA, KX, NX, MB, NB, M, K, N;
    unsigned sA1, sA2, sX1, sX2, sB1, sB2;
    GetIndexBoundaries(matrixLeftStructure_in, lengthLeftX_in, lengthLeftY_in, MA, KA, sA1, sA2);
    GetIndexBoundaries(matrixStructure_in, lengthX_in, lengthY_in, KX, NX, sX1, sX2);
    GetIndexBoundaries(matrixRightStructure_in, lengthRightX_in, lengthRightY_in, MB, NB, sB1, sB2);

    if (MA != MB || KA != KX || NX != NB)
    {
        std::cout << "Error: sizes do not match.\n";
        return;
    }
    else if (MA != KA)
    {
        std::cout << "Error: matrix is not a square matrix.\n";
        return;
    }
    A = matrixLeft_in.pointer;
    X = matrix_out.pointer;
    B = matrixRight_in.pointer;
    M = MA;
    K = KA;
    N = NX;
    for (unsigned j = 0u; j < N; j++)
    {
        for (unsigned k = 0u; k < K; k++)
        {
            unsigned kk = K - 1u - k;
            double sum = 0.0;
            for (unsigned i = k + 1; i < K; i++)
            {
                sum += (*(A + i * sA2 + kk * sA1)) * (*(X + j * sX2 + i * sX1));
            }
            *(X + j * sX2 + kk * sX1) = (1.0 / (*(A + kk * sA2 + kk * sA1))) * (*(B + j * sB2 + kk * sB1) - sum);
        }
    }
}

// Main Methods

void MathCPU::CholeskyDecomposition(Pointer<double> decomposition_out, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in)
{
    if (lengthX_in != lengthY_in)
    {
        std::cout << "Error: dimensions do not match.\n";
        return;
    }
    unsigned length = lengthX_in;
    double *L_out, *A_in;
    L_out = decomposition_out.pointer;
    A_in = matrix_in.pointer;
    for (unsigned k = 0u; k < length * length; k++)
    {
        L_out[k] = 0.0;
    }

    for (unsigned j = 0u; j < length; j++)
    {
        double sum = 0.0;
        for (unsigned k = 0u; k < j; k++)
        {
            sum += L_out[k * length + j] * L_out[k * length + j];
        }
        L_out[j * length + j] = sqrt(A_in[j * length + j] - sum);

        for (unsigned i = j + 1u; i < length; i++)
        {
            sum = 0.0;
            for (unsigned k = 0u; k < j; k++)
            {
                sum += L_out[k * length + i] * L_out[k * length + j];
            }
            L_out[j * length + i] = (1.0 / L_out[j * length + j]) * (A_in[j * length + i] - sum);
        }
    }
}

void MathCPU::CholeskySolver(Pointer<double> X_out, MatrixOperationSide operationSide_in,
                             Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
                             Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in)
{
    Pointer<double> workspace;
    workspace.pointer = new double[lengthAX_in * lengthAY_in];
    workspace.context = PointerContext::CPU_Only;
    workspace.type = PointerType::CPU;
    CholeskyDecomposition(workspace, A_in, lengthAX_in, lengthAY_in);
    switch (operationSide_in)
    {
    case MatrixOperationSide::Left:
        ForwardSubstitution(X_out, MatrixStructure::Natural, lengthAY_in, lengthBY_in,
                            workspace, MatrixStructure::Natural, lengthAX_in, lengthAY_in,
                            B_in, MatrixStructure::Natural, lengthBX_in, lengthBY_in);
        BackwardSubstitution(X_out, MatrixStructure::Natural, lengthAY_in, lengthBY_in,
                            workspace, MatrixStructure::Transposed, lengthAX_in, lengthAY_in,
                            X_out, MatrixStructure::Natural, lengthBX_in, lengthBY_in);
        break;
    case MatrixOperationSide::Right:
        ForwardSubstitution(X_out, MatrixStructure::Transposed, lengthAY_in, lengthBY_in,
                            workspace, MatrixStructure::Transposed, lengthAX_in, lengthAY_in,
                            B_in, MatrixStructure::Transposed, lengthBX_in, lengthBY_in);
        BackwardSubstitution(X_out, MatrixStructure::Transposed, lengthAY_in, lengthBY_in,
                            workspace, MatrixStructure::Natural, lengthAX_in, lengthAY_in,
                            X_out, MatrixStructure::Transposed, lengthBX_in, lengthBY_in);
        break;

    default:
        break;
    }
    delete[] workspace.pointer;
}
// Wrapper Methods
void MathCPU::Decomposition(Pointer<double> decomposition_out, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> pivot_out = Pointer<double>())
{
}

void MathCPU::Solve(Pointer<double> X_out, LinearSolverType solverType_in, MatrixOperationSide operationSide_in,
                    Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
                    Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in)
{
}
