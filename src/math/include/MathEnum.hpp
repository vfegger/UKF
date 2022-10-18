#ifndef MATH_ENUM_HEADER
#define MATH_ENUM_HEADER

enum MatrixStructure{
    MatrixStructure_None,
    MatrixStructure_Natural,
    MatrixStructure_Transposed,
};

enum DecompositionType{
    DecompositionType_Default,
    DecompositionType_LU,
    DecompositionType_LUP,
    DecompositionType_Cholesky,
    DecompositionType_QR,
};

enum LinearSolverType{
    LinearSolverType_Default,
    LinearSolverType_LU,
    LinearSolverType_LUP,
    LinearSolverType_Cholesky,
    LinearSolverType_QR,
    LinearSolverType_GJ,
    LinearSolverType_GS,
    LinearSolverType_SOR,
};

enum MatrixOperationSide{
    MatrixOperationSide_Left,
    MatrixOperationSide_Right,
};

#endif