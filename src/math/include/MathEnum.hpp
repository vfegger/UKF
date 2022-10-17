#ifndef MATH_ENUM_HEADER
#define MATH_ENUM_HEADER

enum MatrixStructure{
    None,Natural,Transposed,
};

enum DecompositionType{
    Default,LU,LUP,Cholesky,QR,
};

enum LinearSolverType{
    Default,LU,LUP,Cholesky,QR,GJ,GS,SOR,
};

enum MatrixOperationSide{
    Left,Right,
};

#endif