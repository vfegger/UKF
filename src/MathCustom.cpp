#include "MathCustom.hpp"
#include <math.h>

double* AddVector(double* a_in, double* b_in, double* c_out, unsigned length){
    for(unsigned i = 0u; i < length; i++){
        c_out[i] = a_in[i] + b_in[i];
    }
    return c_out;
}

double* AddVector(double* a_inout, double* b_in, unsigned length){
    for(unsigned i = 0u; i < length; i++){
        a_inout[i] += b_in[i];
    }
    return a_inout;
}

double* SubVector(double* a_in, double* b_in, double* c_out, unsigned length){
    for(unsigned i = 0u; i < length; i++){
        c_out[i] = a_in[i] - b_in[i];
    }
    return c_out;
}

double* SubVector(double* a_inout, double* b_in, unsigned length){
    for(unsigned i = 0u; i < length; i++){
        a_inout[i] -= b_in[i];
    }
    return a_inout;
}

double* Multiply(double* a_in, double* b_in, double* c_out, unsigned height_out, unsigned width_out, unsigned shared_in){
    for(unsigned i = 0u; i < width_out; i++){
        for(unsigned j = 0u; j < height_out; j++){
            double acc = 0.0;
            for(unsigned k = 0u; k < shared_in; k++){
                acc += a_in[i*shared_in+k]*b_in[k*width_out+j];
            }
             c_out[i*height_out+j] = acc;
        }
    }
    return c_out;
}

double* MultiplyTransposed(double* a_in, double* b_in, double* c_out, unsigned height_out, unsigned width_out, unsigned shared_in){
    for(unsigned i = 0u; i < width_out; i++){
        for(unsigned j = 0u; j < height_out; j++){
            double acc = 0.0;
            for(unsigned k = 0u; k < shared_in; k++){
                acc += a_in[i*shared_in+k]*b_in[j*shared_in+k];
            }
             c_out[i*height_out+j] = acc;
        }
    }
    return c_out;
}

double* CholeskyDecomposition(double* a_in, double* b_out, unsigned height, unsigned width){
    unsigned diagDim = (width > height) ? width : height;
    for (unsigned k = 0u; k < diagDim; k++)
    {
        double acc = 0.0;
        for(unsigned j = 0u; j < k; j++){
            acc += b_out[j*height+k]*b_out[j*height+k];
        }
        b_out[k*height+k] = sqrt(a_in[k*height+k]- acc);
        for(unsigned i = k + 1u; i < height; i++){
            acc = 0.0;
            for(unsigned j = 0u; j < k; j++){
                acc += b_out[j*height+k]*b_out[j*height+i];
            }
            b_out[k*height+i] = (1.0/b_out[k*height+k])*(a_in[k*height+i]-acc);
        }
    }
    return b_out;
}

double* Identity(double* a_inout, unsigned height, unsigned width){
    for(unsigned i = 0u; i < width; i++){
        for(unsigned j = 0u; j < height; j++){
            a_inout[i*height+j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return a_inout;
}

double* Transpose(double* a_in, double* b_out, unsigned height, unsigned width){
    for(unsigned i = 0u; i < width; i++){
        for(unsigned j = 0u; j < height; j++){
            b_out[j*width+i] = a_in[i*height+j];
        }
    }
    return b_out;
}

double* ForwardSubstituition(double* a_in, double* x_inout, double* b_in, unsigned height, unsigned width, unsigned width_RHS){
    for(unsigned k = 0u; k < width_RHS; k++){
        for(unsigned i = 0u; i < height; i++){
            double acc = b_in[k*height+i];
            for(unsigned j = 0; j < i; j++){
                acc -= a_in[j*height+i]*x_inout[k*height+j];
            }
            x_inout[k*height+i] = acc/a_in[i*height+i];
        }
    }
    return x_inout;
}

double* BackwardSubstituition(double* a_in, double* x_inout, double* b_in, unsigned height, unsigned width, unsigned width_RHS){
    unsigned ii;
    for(unsigned k = 0u; k < width_RHS; k++){
        for(unsigned i = 0u; i < height; i++){
            ii = height - i - 1u;
            double acc = b_in[k*height+ii];
            for(unsigned j = ii; j < width; j++){
                acc -= a_in[j*height+i]*x_inout[k*height+j];
            }
            x_inout[k*height+i] = acc/a_in[i*height+i];
        }
    }
    return x_inout;
}

double* PseudoInverse(double* a_in, double* b_out, unsigned height, unsigned width){
    double* aux = new double[width*width];
    double* aux1 = new double[width*width];
    double* decomposition = new double[width*width];
    double* decompositionT = new double[width*width];
    CholeskyDecomposition(MultiplyTransposed(a_in,a_in,aux,width,width,height),decomposition,width,width);
    Transpose(decomposition,decompositionT,width,width);
    Identity(aux,width,width);
    BackwardSubstituition(decomposition,aux1,aux,width,width,width);
    ForwardSubstituition(decompositionT,aux,aux1,width,width,width);
    MultiplyTransposed(aux,a_in,b_out,width,height,width);
}

double* Solver();
double* RightSolver();
