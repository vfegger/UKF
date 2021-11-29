#ifndef MATH_CUSTOM_HEADER
#define MATH_CUSTOM_HEADER

#include <math.h>

double* AddVector(double* a_in, double* b_in, double* c_out, unsigned length);
double* AddVector(double* a_inout, double* b_in, unsigned length);
double* SubVector(double* a_in, double* b_in, double* c_out, unsigned length);
double* SubVector(double* a_inout, double* b_in, unsigned length);
double* Multiply(double* a_in, double* b_in, double* c_out, unsigned height_out, unsigned width_out, unsigned shared_in);
double* MultiplyTransposed(double* a_in, double* b_in, double* c_out, unsigned height_out, unsigned width_out, unsigned shared_in);
double* CholeskyDecomposition(double* a_in, double* b_out, unsigned height, unsigned width);
// TODO: Create the functions below
double* PseudoInverse(double* a_in, double* b_out, unsigned height, unsigned width);
double* Transpose();
double* Solver();
double* RightSolver();

#endif