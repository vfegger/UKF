#include "../include/Math.hpp"

// Error Check
bool CheckType(PointerType typeLeft, PointerType typeRight)
{
    if (typeLeft != typeRight)
    {
        std::cout << "Error: Pointer types do not match.\n";
        return false;
    }
    return true;
}

// In-Placed Calculation

// Vector Element-wise Addition
void Math::Add(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in)
{
    if (!CheckType(vector_inout.type, vector_in.type))
    {
        return;
    }
    switch (vector_inout.type)
    {
    case PointerType::CPU:
        MathCPU::Add(vector_inout, vector_in, length_in);
        break;
    case PointerType::GPU:
        MathGPU::Add(vector_inout, vector_in, length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

// Vector Element-wise Subtraction
void Math::Sub(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in)
{
    if (!CheckType(vector_inout.type, vector_in.type))
    {
        return;
    }
    switch (vector_inout.type)
    {
    case PointerType::CPU:
        MathCPU::Sub(vector_inout, vector_in, length_in);
        break;
    case PointerType::GPU:
        MathGPU::Sub(vector_inout, vector_in, length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

// Vector Constant Multiplication
void Math::Mul(Pointer<double> vector_inout, double value_in, unsigned length_in)
{
    switch (vector_inout.type)
    {
    case PointerType::CPU:
        MathCPU::Mul(vector_inout, value_in, length_in);
        break;
    case PointerType::GPU:
        MathGPU::Mul(vector_inout, value_in, length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

// Vector Element-wise Multiplication
void Math::Mul(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in)
{
    if (!CheckType(vector_inout.type, vector_in.type))
    {
        return;
    }
    switch (vector_inout.type)
    {
    case PointerType::CPU:
        MathCPU::Mul(vector_inout, vector_in, length_in);
        break;
    case PointerType::GPU:
        MathGPU::Mul(vector_inout, vector_in, length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

// Out-Placed Calculation

// Vector Element-wise Addition
void Math::Add(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    if ((!CheckType(vector_out.type, vectorLeft_in.type)) || (!CheckType(vector_out.type, vectorRight_in.type)))
    {
        return;
    }
    switch (vector_out.type)
    {
    case PointerType::CPU:
        MathCPU::Add(vector_out, vectorLeft_in, vectorRight_in, length_in);
        break;
    case PointerType::GPU:
        MathGPU::Add(vector_out, vectorLeft_in, vectorRight_in, length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}
// Vector Element-wise Subtraction
void Math::Sub(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    if ((!CheckType(vector_out.type, vectorLeft_in.type)) || (!CheckType(vector_out.type, vectorRight_in.type)))
    {
        return;
    }
    switch (vector_out.type)
    {
    case PointerType::CPU:
        MathCPU::Sub(vector_out, vectorLeft_in, vectorRight_in, length_in);
        break;
    case PointerType::GPU:
        MathGPU::Sub(vector_out, vectorLeft_in, vectorRight_in, length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}
// Vector Constant Multiplication
void Math::Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in)
{
    if (!CheckType(vector_out.type, vectorLeft_in.type))
    {
        return;
    }
    switch (vector_out.type)
    {
    case PointerType::CPU:
        MathCPU::Mul(vector_out, vectorLeft_in, valueRight_in, length_in);
        break;
    case PointerType::GPU:
        MathGPU::Mul(vector_out, vectorLeft_in, valueRight_in, length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}
// Vector Element-wise Multiplication
void Math::Mul(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in)
{
    if ((!CheckType(vector_out.type, vectorLeft_in.type)) || (!CheckType(vector_out.type, vectorRight_in.type)))
    {
        return;
    }
    switch (vector_out.type)
    {
    case PointerType::CPU:
        MathCPU::Mul(vector_out, vectorLeft_in, vectorRight_in, length_in);
        break;
    case PointerType::GPU:
        MathGPU::Mul(vector_out, vectorLeft_in, vectorRight_in, length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

// Matrix Multiplication TODO
void Math::MatrixMultiplication(Pointer<double> matrix_out,
                                Pointer<double> matrixLeft_in, MatrixStructure matrixLeftStructure_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
                                Pointer<double> matrixRight_in, MatrixStructure matrixRightStructure_in, unsigned lengthRightX_in, unsigned lengthRightY_in)
{
    if ((!CheckType(matrix_out.type, matrixLeft_in.type)) || (!CheckType(matrix_out.type, matrixRight_in.type)))
    {
        return;
    }
    switch (matrix_out.type)
    {
    case PointerType::CPU:
        MathCPU::MatrixMultiplication(matrix_out, matrixLeft_in, matrixLeftStructure_in, lengthLeftX_in, lengthLeftY_in, matrixRight_in, matrixRightStructure_in, lengthRightX_in, lengthRightY_in);
        break;
    case PointerType::GPU:
        MathGPU::MatrixMultiplication(matrix_out, matrixLeft_in, matrixLeftStructure_in, lengthLeftX_in, lengthLeftY_in, matrixRight_in, matrixRightStructure_in, lengthRightX_in, lengthRightY_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

// Operators

// In-Place Operator Distribution
void Math::Operation(void (*operation_in)(Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in),
                     Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                     unsigned strideOut_in, unsigned strideLeft_in, unsigned strideRight_in,
                     unsigned offsetOut_in, unsigned offsetLeft_in, unsigned offsetRight_in)
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
void Math::Operation(void (*operation_in)(Pointer<double> vector_out, Pointer<double> vectorLeft_in, double valueRight_in, unsigned length_in),
                     Pointer<double> vector_out, Pointer<double> vectorLeft_in, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                     unsigned strideOut_in, unsigned strideLeft_in, unsigned strideRight_in,
                     unsigned offsetOut_in, unsigned offsetLeft_in, unsigned offsetRight_in)
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
void Math::Operation(void (*operation_in)(Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in),
                     Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                     unsigned strideLeft_in, unsigned strideRight_in,
                     unsigned offsetLeft_in, unsigned offsetRight_in)
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
void Math::Operation(void (*operation_in)(Pointer<double> vector_inout, double valueRight_in, unsigned length_in),
                     Pointer<double> vector_inout, Pointer<double> vectorRight_in, unsigned length_in, unsigned iteration_in,
                     unsigned strideLeft_in, unsigned strideRight_in,
                     unsigned offsetLeft_in, unsigned offsetRight_in)
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
void Math::Mean(Pointer<double> vector_out, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> weight_in)
{
    if ((!CheckType(vector_out.type, matrix_in.type)) || (weight_in.pointer != NULL && (!CheckType(vector_out.type, weight_in.type))))
    {
        return;
    }
    switch (vector_out.type)
    {
    case PointerType::CPU:
        MathCPU::Mean(vector_out, matrix_in, lengthX_in, lengthY_in, weight_in);
        break;
    case PointerType::GPU:
        MathGPU::Mean(vector_out, matrix_in, lengthX_in, lengthY_in, weight_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

// Linear System Solvers

// Helper Functions

// Wrapper Methods
void Math::Decomposition(Pointer<double> decomposition_out, DecompositionType decompositionType_in, Pointer<double> matrix_in, unsigned lengthX_in, unsigned lengthY_in, Pointer<double> pivot_out)
{
    if ((!CheckType(decomposition_out.type, matrix_in.type)) || (pivot_out.pointer != NULL && (!CheckType(decomposition_out.type, pivot_out.type))))
    {
        return;
    }
    switch (decomposition_out.type)
    {
    case PointerType::CPU:
        MathCPU::Decomposition(decomposition_out, decompositionType_in, matrix_in, lengthX_in, lengthY_in, pivot_out);
        break;
    case PointerType::GPU:
        MathGPU::Decomposition(decomposition_out, decompositionType_in, matrix_in, lengthX_in, lengthY_in, pivot_out);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

void Math::Solve(Pointer<double> X_out, LinearSolverType solverType_in, MatrixOperationSide operationSide_in,
                 Pointer<double> A_in, unsigned lengthAX_in, unsigned lengthAY_in,
                 Pointer<double> B_in, unsigned lengthBX_in, unsigned lengthBY_in)
{
    if ((!CheckType(X_out.type, A_in.type)) || (!CheckType(X_out.type, B_in.type)))
    {
        return;
    }
    switch (X_out.type)
    {
    case PointerType::CPU:
        MathCPU::Solve(X_out, solverType_in, operationSide_in, A_in, lengthAX_in, lengthAY_in, B_in, lengthBX_in, lengthBY_in);
        break;
    case PointerType::GPU:
        MathGPU::Solve(X_out, solverType_in, operationSide_in, A_in, lengthAX_in, lengthAY_in, B_in, lengthBX_in, lengthBY_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}
