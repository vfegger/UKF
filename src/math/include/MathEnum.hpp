#ifndef MATH_ENUM_HEADER
#define MATH_ENUM_HEADER

enum MatrixStructure{
    None,Natural,Transposed,
};

enum LinearSolverType{
    Default,LU,LUP,QR,GJ,GS,SOR,
};

enum MatrixOperationSide{
    Left,Right,
};

#endif