#include "../include/MathCPU.hpp"

// Auxiliary Functions
void MathCPU::Print(Pointer<double> vector_in, unsigned length_in)
{
    std::cout << "Vector -> Size : " << length_in << "\n";
    for (unsigned i = 0u; i < length_in; i++)
    {
        std::cout << " " << vector_in.pointer[i] << "\n";
    }
}

void MathCPU::Print(Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in)
{
    std::cout << "Matrix -> Size : " << lengthX_in << ":" << lengthY_in << "\n";
    std::cout << std::scientific;
    for (unsigned i = 0u; i < lengthX_in; i++)
    {
        for (unsigned j = 0u; j < lengthY_in; j++)
        {
            std::cout << " " << matrix_in.pointer[j * lengthX_in + i];
        }
        std::cout << "\n";
    }
}

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
// Matrix Multiplication
template <typename T>
void Swap(T &left_inout, T &right_inout)
{
    T aux = left_inout;
    left_inout = right_inout;
    right_inout = aux;
}

void GetIndexBoundaries(MatrixStructure matrixStructure_in, unsigned lengthX_in, unsigned lengthY_in, unsigned &sizeX_out, unsigned &sizeY_out, unsigned &strideX_out, unsigned &strideY_out)
{
    switch (matrixStructure_in)
    {
    case MatrixStructure_Natural:
        sizeX_out = lengthX_in;
        sizeY_out = lengthY_in;
        strideX_out = 1u;
        strideY_out = lengthX_in;
        break;
    case MatrixStructure_Transposed:
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

void GetIndexOutBoundaries(MatrixStructure structO_in, unsigned lengthOX_in, unsigned lengthOY_in, unsigned &sizeOX_out, unsigned &sizeOY_out, unsigned &strideOX_out, unsigned &strideOY_out,
                           double *&L_inout, unsigned &sizeLX_out, unsigned &sizeLY_out, unsigned &strideLX_out, unsigned &strideLY_out,
                           double *&R_inout, unsigned &sizeRX_out, unsigned &sizeRY_out, unsigned &strideRX_out, unsigned &strideRY_out)
{
    GetIndexBoundaries(structO_in, lengthOX_in, lengthOY_in, sizeOX_out, sizeOY_out, strideOX_out, strideOY_out);
    if (structO_in == MatrixStructure_Transposed)
    {
        Swap(L_inout, R_inout);
        Swap(sizeLX_out, sizeRY_out);
        Swap(sizeLY_out, sizeRX_out);
        Swap(strideLX_out, strideRY_out);
        Swap(strideLY_out, strideRX_out);
    }
}

void MathCPU::MatrixMultiplication(double alpha,
                                   Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                                   Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
                                   double beta,
                                   Pointer<double> matrix_out, MatrixStructure matrixOutStructure_in, unsigned lengthOutX_in, unsigned lengthOutY_in,
                                   Pointer<double> weight_in)
{
    double *pA, *pB, *pC;
    unsigned ML, NR, KL, KR, M, N, K;
    unsigned sA1, sB1, sC1;
    unsigned sA2, sB2, sC2;
    pA = matrixLeft_in.pointer;
    pB = matrixRight_in.pointer;
    pC = matrix_out.pointer;
    GetIndexBoundaries(matrixLeftStructure_in, lengthLeftX_in, lengthLeftY_in, ML, KL, sA1, sA2);
    GetIndexBoundaries(matrixRightStructure_in, lengthRightX_in, lengthRightY_in, KR, NR, sB1, sB2);
    GetIndexOutBoundaries(matrixOutStructure_in, lengthOutX_in, lengthOutY_in, M, N, sC1, sC2,
                          pA, ML, KL, sA1, sA2, pB, KR, NR, sB1, sB2);
    if (KL == KR && M == ML && N == NR)
    {
        K = KL;
    }
    else
    {
        std::cout << "Error: Sizes do not match.\n";
        std::cout << "M: " << ML << " " << M << "\n";
        std::cout << "N: " << NR << " " << N << "\n";
        std::cout << "K: " << KL << " " << KR << "\n";
        return;
    }
    double auxAlpha = 0.0;
    double auxBeta = 0.0;
    double *aux = pC;
    if (matrixOutStructure_in == MatrixStructure_Transposed)
    {
        aux = new double[M * N];
        for (unsigned j = 0; j < N; j++)
        {
            for (unsigned i = 0; i < M; i++)
            {
                aux[j * M + i] = pC[i * N + j];
            }
        }
    }
    if (weight_in.pointer == NULL)
    {
        for (unsigned j = 0; j < N; j++)
        {
            for (unsigned i = 0; i < M; i++)
            {
                auxAlpha = 0.0;
                auxBeta = aux[j * M + i];
                for (unsigned k = 0; k < K; k++)
                {
                    auxAlpha += pA[k * sA2 + i * sA1] * pB[j * sB2 + k * sB1];
                }
                pC[j * M + i] = alpha * auxAlpha + beta * auxBeta;
            }
        }
    }
    else
    {
        for (unsigned j = 0; j < N; j++)
        {
            for (unsigned i = 0; i < M; i++)
            {
                auxAlpha = 0.0;
                auxBeta = aux[j * M + i];
                for (unsigned k = 0; k < K; k++)
                {
                    auxAlpha += weight_in.pointer[k] * pA[k * sA2 + i * sA1] * pB[j * sB2 + k * sB1];
                }
                pC[j * M + i] = alpha * auxAlpha + beta * auxBeta;
            }
        }
    }
    if (matrixOutStructure_in == MatrixStructure_Transposed)
    {
        delete[] aux;
    }
}

// Reducibles Operations
void MathCPU::Mean(Pointer<double> vector_out, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> weight_in)
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
                vector_out.pointer[i] += matrix_in.pointer[j * lengthX_in + i];
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
                vector_out.pointer[i] += matrix_in.pointer[j * lengthX_in + i] * weight_in.pointer[j];
            }
        }
    }
}

bool MathCPU::Compare(Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    for (unsigned i = 0u; i < length_in; i++)
    {
        if (vectorLeft_in.pointer[i] != vectorRight_in.pointer[i])
        {
            return false;
        }
    }
    return true;
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
                sum += A[i * sA2 + k * sA1] * X[j * sX2 + i * sX1];
            }
            X[j * sX2 + k * sX1] = (1.0 / A[k * sA2 + k * sA1]) * (B[j * sB2 + k * sB1] - sum);
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
            for (unsigned i = kk + 1; i < K; i++)
            {
                sum += A[i * sA2 + kk * sA1] * X[j * sX2 + i * sX1];
            }
            X[j * sX2 + kk * sX1] = (1.0 / A[kk * sA2 + kk * sA1]) * (B[j * sB2 + kk * sB1] - sum);
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

void MathCPU::CholeskySolver(Pointer<double> X_out,
                             Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
                             Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in)
{
    Pointer<double> workspace;
    workspace.pointer = new double[lengthAX_in * lengthAY_in];
    workspace.context = PointerContext::CPU_Only;
    workspace.type = PointerType::CPU;
    CholeskyDecomposition(workspace, A_in, lengthAX_in, lengthAY_in);
    ForwardSubstitution(X_out, MatrixStructure_Natural, lengthAY_in, lengthBY_in,
                        workspace, MatrixStructure_Natural, lengthAX_in, lengthAY_in,
                        B_in, MatrixStructure_Natural, lengthBX_in, lengthBY_in);
    BackwardSubstitution(X_out, MatrixStructure_Natural, lengthAY_in, lengthBY_in,
                         workspace, MatrixStructure_Transposed, lengthAX_in, lengthAY_in,
                         X_out, MatrixStructure_Natural, lengthBX_in, lengthBY_in);
    delete[] workspace.pointer;
}
// Wrapper Methods
void MathCPU::Decomposition(Pointer<double> decomposition_out, DecompositionType decompositionType_in, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> pivot_out)
{
    switch (decompositionType_in)
    {
    case DecompositionType_Cholesky:
        CholeskyDecomposition(decomposition_out, matrix_in, lengthX_in, lengthY_in);
        break;
    default:
        break;
    }
    return;
}

void MathCPU::Solve(Pointer<double> X_out, LinearSolverType solverType_in,
                    Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
                    Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in)
{
    switch (solverType_in)
    {
    case LinearSolverType_Cholesky:
        CholeskySolver(X_out, A_in, lengthAX_in, lengthAY_in, B_in, lengthBX_in, lengthBY_in);
        break;
    default:
        break;
    }
    return;
}
