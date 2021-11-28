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