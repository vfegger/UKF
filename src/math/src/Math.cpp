#include "../include/Math.hpp"

namespace Math {
    void CholeskyDecomposition(double* matrix_out, double* matrix_in, unsigned lengthX_in, unsigned lengthY_in){
        unsigned diagDim = (lengthX_in > lengthY_in) ? lengthX_in : lengthY_in;
        for (unsigned k = 0u; k < diagDim; k++)
        {
            double acc = 0.0;
            for(unsigned j = 0u; j < k; j++){
                acc += matrix_out[j*lengthX_in+k]*matrix_out[j*lengthX_in+k];
            }
            matrix_out[k*lengthX_in+k] = sqrt(matrix_in[k*lengthX_in+k]-acc);
            for(unsigned i = k + 1u; i < lengthX_in; i++){
                acc = 0.0;
                for(unsigned j = 0u; j < k; j++){
                    acc += matrix_out[j*lengthX_in+k]*matrix_out[j*lengthX_in+i];
                }
                matrix_out[k*lengthX_in+i] = (1.0/matrix_out[k*lengthX_in+k])*(matrix_in[k*lengthX_in+i]-acc);
            }
        }
    }
    void AddInPlace(double* matrix_inout, double* matrix_in, unsigned length_in){
        for(unsigned i = 0u; i < length_in){
            matrix_inout[i] += matrix_in[i];
        }
    }
    void SubInPlace(double* matrix_inout, double* matrix_in, unsigned length_in){
        for(unsigned i = 0u; i < length_in){
            matrix_inout[i] -= matrix_in[i];
        }
    }
    
    void MatrixMultiplicationNN(double* matrix_out,
    double* matrixLeft_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in) {
        if(lengthLeftY_in != lengthRightX_in) { 
            std::cout << "Error: Multiplication sizes are not compatible.\n";
        }
        unsigned lengthK = lengthLeftY_in;
        if( weight_in != NULL){
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftX_in; i++){
                    double acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[k*lengthLeftX_in+i] * matrixRight_in[j*lengthRightX_in+k] * weight_in[k];
                    }
                    matrix_out[j*lengthLeftX_in + i] = acc;
                }
            }
        } else {
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftX_in; i++){
                    double acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[k*lengthLeftX_in+i] * matrixRight_in[j*lengthRightX_in+k];
                    }
                    matrix_out[j*lengthLeftX_in + i] = acc;
                }
            }
        }
    }
    void MatrixMultiplicationNT(double* matrix_out,
    double* matrixLeft_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in) {
        if(lengthLeftY_in != lengthRightY_in) { 
            std::cout << "Error: Multiplication sizes are not compatible.\n";
        }
        unsigned lengthK = lengthLeftY_in;
        if(weight_in != NULL){
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftX_in; i++){
                    double acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[k*lengthLeftX_in+i] * matrixRight_in[k*lengthRightY_in+j] * weight_in[k];
                    }
                    matrix_out[j*lengthLeftX_in + i] = acc;
                }
            }
        } else { 
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftX_in; i++){
                    double acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[k*lengthLeftX_in+i] * matrixRight_in[k*lengthRightY_in+j];
                    }
                    matrix_out[j*lengthLeftX_in + i] = acc;
                }
            }
        }
    }
    void MatrixMultiplicationTN(double* matrix_out,
    double* matrixLeft_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in) {
        if(lengthLeftX_in != lengthRightX_in) { 
            std::cout << "Error: Multiplication sizes are not compatible.\n";
        }
        unsigned lengthK = lengthLeftX_in;
        if(weight_in != NULL){
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftY_in; i++){
                    double acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[i*lengthLeftY_in+k] * matrixRight_in[j*lengthRightX_in+k] * weight_in[k];
                    }
                    matrix_out[j*lengthLeftY_in + i] = acc;
                }
            }
        } else {
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftY_in; i++){
                    double acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[i*lengthLeftY_in+k] * matrixRight_in[j*lengthRightX_in+k];
                    }
                    matrix_out[j*lengthLeftY_in + i] = acc;
                }
            }
        }
    }
    void MatrixMultiplicationTT(double* matrix_out,
    double* matrixLeft_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in) {
        if(lengthLeftX_in != lengthRightY_in) { 
            std::cout << "Error: Multiplication sizes are not compatible.\n";
        }
        unsigned lengthK = lengthLeftX_in;
        if(weight_in != NULL){
            for(unsigned j = 0u; j < lengthRightX_in; j++){
                for(unsigned i = 0u; i < lengthLeftY_in; i++){
                    double acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[i*lengthLeftY_in+k] * matrixRight_in[k*lengthRightY_in+j] * weight_in[k];
                    }
                    matrix_out[j*lengthLeftY_in + i] = acc;
                }
            }
        } else {
            for(unsigned j = 0u; j < lengthRightX_in; j++){
                for(unsigned i = 0u; i < lengthLeftY_in; i++){
                    double acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[i*lengthLeftY_in+k] * matrixRight_in[k*lengthRightY_in+j];
                    }
                    matrix_out[j*lengthLeftY_in + i] = acc;
                }
            }
        }
    }
    void MatrixMultiplication(double* matrix_out,
    double* matrixLeft_in, MatrixStructure structureLeft, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, MatrixStructure structureRight, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in){
        if(structureLeft == MatrixStructure::Natural && structureRight == MatrixStructure::Natural){
            MatrixMultiplicationNN(matrix_out,matrixLeft_in,lengthLeftX_in,lengthLeftY_in,matrixRight_in, lengthRightX_in, lengthRightY_in, weight_in);
        } else if (structureLeft == MatrixStructure::Natural){
            MatrixMultiplicationNT(matrix_out,matrixLeft_in,lengthLeftX_in,lengthLeftY_in,matrixRight_in, lengthRightX_in, lengthRightY_in, weight_in);
        } else if (structureRight == MatrixStructure::Natural){
            MatrixMultiplicationTN(matrix_out,matrixLeft_in,lengthLeftX_in,lengthLeftY_in,matrixRight_in, lengthRightX_in, lengthRightY_in, weight_in);
        } else {
            MatrixMultiplicationTT(matrix_out,matrixLeft_in,lengthLeftX_in,lengthLeftY_in,matrixRight_in, lengthRightX_in, lengthRightY_in, weight_in);
        }
    }
    void DistributeOperation(void (*f)(double* matrixLeft_inout, double* matrixRight_in, unsigned length_in), double* matrixLeft_inout, double* matrixRight_in, unsigned length_in, unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in, unsigned offsetLeft_in, unsigned offsetRight_in){
        double* left = matrixLeft_inout + offsetLeft_in;
        double* right = matrixRight_in + offsetRight_in;
        if(length_in < strideLeft_in){
            std::cout << "Warning: The behavior is undefined to left strides lower than the length of the operation.\n";
        }
        for(unsigned i = 0u; i < iteration_in; i++) {
            f(matrixLeft_inout + i * strideLeft_in,matrixRight_in + i * strideRight_in,length_in);
        }
    }
    void Mean(double* vector_out, double* matrix_in, unsigned lengthX_in, unsigned lengthY_in, double* weight_in){
        for(unsigned i = 0u; i < lengthX_in; i++){
            vector_out[i] = 0.0;
        }
        if(weight_in != NULL){
            for(unsigned j = 0u; j < lengthY_in; j++){
                for(unsigned i = 0u; i < lengthX_in; i++){
                    vector_out[i] += matrix_in[j*lengthX_in+i] * weight_in[j];
                }
            }
        } else {
            for(unsigned j = 0u; j < lengthY_in; j++){
                for(unsigned i = 0u; i < lengthX_in; i++){
                    vector_out[i] += matrix_in[j*lengthX_in+i];
                }
            }
        }
    }
}
