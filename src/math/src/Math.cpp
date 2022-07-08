#include "../include/Math.hpp"

namespace Math {
    void PrintMatrix(double* matrix_in, unsigned lengthX_in, unsigned lengthY_in){
        //std::cout << "Size X = " << lengthX_in << "; ";
        //std::cout << "Size Y = " << lengthY_in << "\n";

        for(unsigned i = 0u; i < lengthX_in; i++){
            for(unsigned j = 0u; j < lengthY_in; j++){
                std::cout << std::scientific << std::fixed << std::setprecision(6) << matrix_in[j*lengthX_in+i] << " "; 
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

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
        for(unsigned i = 0u; i < length_in; i++){
            matrix_inout[i] += matrix_in[i];
        }
    }
    void SubInPlace(double* matrix_inout, double* matrix_in, unsigned length_in){
        for(unsigned i = 0u; i < length_in; i++){
            matrix_inout[i] -= matrix_in[i];
        }
    }
    void ConstantMultiplicationInPlace(double* matrix_inout, double value_in, unsigned length_in){
        for(unsigned i = 0u; i < length_in; i++){
            matrix_inout[i] *= value_in;
        }
    }
    void MatrixMultiplicationNN(double* matrix_out, double alpha, double beta,
    double* matrixLeft_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in) {
        if(lengthLeftY_in != lengthRightX_in) { 
            std::cout << "Error: Multiplication sizes are not compatible.\n";
        }
        unsigned lengthK = lengthLeftY_in;
        double aux, acc;
        if( weight_in != NULL){
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftX_in; i++){
                    aux = beta * matrix_out[j*lengthLeftX_in + i];
                    acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[k*lengthLeftX_in+i] * matrixRight_in[j*lengthRightX_in+k] * weight_in[k];
                    }
                    matrix_out[j*lengthLeftX_in + i] = aux + alpha * acc;
                }
            }
        } else {
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftX_in; i++){
                    aux = beta * matrix_out[j*lengthLeftX_in + i];
                    acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[k*lengthLeftX_in+i] * matrixRight_in[j*lengthRightX_in+k];
                    }
                    matrix_out[j*lengthLeftX_in + i] = aux + alpha * acc;
                }
            }
        }
    }
    void MatrixMultiplicationNT(double* matrix_out, double alpha, double beta,
    double* matrixLeft_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in) {
        if(lengthLeftY_in != lengthRightY_in) { 
            std::cout << "Error: Multiplication sizes are not compatible.\n";
        }
        unsigned lengthK = lengthLeftY_in;
        double aux, acc;
        if(weight_in != NULL){
            for(unsigned j = 0u; j < lengthRightX_in; j++){
                for(unsigned i = 0u; i < lengthLeftX_in; i++){
                    aux = beta * matrix_out[j*lengthLeftX_in + i];
                    acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[k*lengthLeftX_in+i] * matrixRight_in[k*lengthRightX_in+j] * weight_in[k];
                    }
                    matrix_out[j*lengthLeftX_in + i] = aux + alpha * acc;
                }
            }
        } else { 
            for(unsigned j = 0u; j < lengthRightX_in; j++){
                for(unsigned i = 0u; i < lengthLeftX_in; i++){
                    aux = beta * matrix_out[j*lengthLeftX_in + i];
                    acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[k*lengthLeftX_in+i] * matrixRight_in[k*lengthRightX_in+j];
                    }
                    matrix_out[j*lengthLeftX_in + i] = aux + alpha * acc;
                }
            }
        }
    }
    void MatrixMultiplicationTN(double* matrix_out, double alpha, double beta,
    double* matrixLeft_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in) {
        if(lengthLeftX_in != lengthRightX_in) { 
            std::cout << "Error: Multiplication sizes are not compatible.\n";
        }
        unsigned lengthK = lengthLeftX_in;
        double aux, acc;
        if(weight_in != NULL){
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftY_in; i++){
                    aux = beta * matrix_out[j*lengthLeftY_in + i];
                    acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[i*lengthLeftX_in+k] * matrixRight_in[j*lengthRightX_in+k] * weight_in[k];
                    }
                    matrix_out[j*lengthLeftY_in + i] = aux + alpha * acc;
                }
            }
        } else {
            for(unsigned j = 0u; j < lengthRightY_in; j++){
                for(unsigned i = 0u; i < lengthLeftY_in; i++){
                    aux = beta * matrix_out[j*lengthLeftY_in + i];
                    acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[i*lengthLeftX_in+k] * matrixRight_in[j*lengthRightX_in+k];
                    }
                    matrix_out[j*lengthLeftY_in + i] = aux + alpha * acc;
                }
            }
        }
    }
    void MatrixMultiplicationTT(double* matrix_out, double alpha, double beta,
    double* matrixLeft_in, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in) {
        if(lengthLeftX_in != lengthRightY_in) { 
            std::cout << "Error: Multiplication sizes are not compatible.\n";
        }
        unsigned lengthK = lengthLeftX_in;
        double aux, acc;
        if(weight_in != NULL){
            for(unsigned j = 0u; j < lengthRightX_in; j++){
                for(unsigned i = 0u; i < lengthLeftY_in; i++){
                    aux = beta * matrix_out[j*lengthLeftY_in + i];
                    acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[i*lengthLeftX_in+k] * matrixRight_in[k*lengthRightX_in+j] * weight_in[k];
                    }
                    matrix_out[j*lengthLeftY_in + i] = aux + alpha * acc;
                }
            }
        } else {
            for(unsigned j = 0u; j < lengthRightX_in; j++){
                for(unsigned i = 0u; i < lengthLeftY_in; i++){
                    aux = beta * matrix_out[j*lengthLeftY_in + i];
                    acc = 0.0;
                    for(unsigned k = 0u; k < lengthK; k++){
                        acc += matrixLeft_in[i*lengthLeftX_in+k] * matrixRight_in[k*lengthRightX_in+j];
                    }
                    matrix_out[j*lengthLeftY_in + i] = aux + alpha * acc;
                }
            }
        }
    }
    void MatrixMultiplication(double* matrix_out, double alpha, double beta,
    double* matrixLeft_in, MatrixStructure structureLeft, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, MatrixStructure structureRight, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in){
        if(structureLeft == MatrixStructure::Natural && structureRight == MatrixStructure::Natural){
            MatrixMultiplicationNN(matrix_out,alpha,beta,matrixLeft_in,lengthLeftX_in,lengthLeftY_in,matrixRight_in, lengthRightX_in, lengthRightY_in, weight_in);
        } else if (structureLeft == MatrixStructure::Natural){
            MatrixMultiplicationNT(matrix_out,alpha,beta,matrixLeft_in,lengthLeftX_in,lengthLeftY_in,matrixRight_in, lengthRightX_in, lengthRightY_in, weight_in);
        } else if (structureRight == MatrixStructure::Natural){
            MatrixMultiplicationTN(matrix_out,alpha,beta,matrixLeft_in,lengthLeftX_in,lengthLeftY_in,matrixRight_in, lengthRightX_in, lengthRightY_in, weight_in);
        } else {
            MatrixMultiplicationTT(matrix_out,alpha,beta,matrixLeft_in,lengthLeftX_in,lengthLeftY_in,matrixRight_in, lengthRightX_in, lengthRightY_in, weight_in);
        }
    }
    void DistributeOperation(void (*f)(double* matrixLeft_inout, double* matrixRight_in, unsigned length_in), double* matrixLeft_inout, double* matrixRight_in, unsigned length_in, unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in, unsigned offsetLeft_in, unsigned offsetRight_in){
        double* left = matrixLeft_inout + offsetLeft_in;
        double* right = matrixRight_in + offsetRight_in;
        if(length_in < strideLeft_in){
            std::cout << "Warning: The behavior is undefined to left strides lower than the length of the operation.\n";
        }
        for(unsigned i = 0u; i < iteration_in; i++) {
            f(left + i * strideLeft_in,right + i * strideRight_in,length_in);
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

    void RHGaussSeidelIteration(double* X_out, double* A, double* B, unsigned lengthX_in, unsigned lengthY_in){
        for(unsigned i = 0u; i < lengthX_in; i++){
            for(unsigned j = 0u; j < lengthY_in; j++){
                double acc = 0.0;
                for(unsigned k = 0u; k < j; k++){
                    acc += X_out[k*lengthX_in+i]*A[j*lengthY_in+k];
                }                
                for(unsigned k = j+1; k < lengthY_in; k++){
                    acc += X_out[k*lengthX_in+i]*A[j*lengthY_in+k];
                }
                X_out[j*lengthX_in+i] = (1.0/A[j*lengthY_in+j])*(B[j*lengthX_in+i]-acc);
            }
        }
    }

    double dist(double* a, double* b, unsigned length_in){
        double result = 0.0;
        for(unsigned i = 0u; i < length_in; i++){
            result += (a[i]-b[i])*(a[i]-b[i]);
        }
        return result;
    }

    void RHSolver(double* X_out, double* A, double* B, unsigned lengthX_in, unsigned lengthY_in, double tol){
        for(unsigned j = 0u; j < lengthY_in; j++){
            for(unsigned i = 0u; i < lengthX_in; i++){
                X_out[j*lengthX_in+i] = B[j*lengthX_in+i];
            }
        }
        double* X_aux = new double[lengthX_in*lengthY_in];
        double tol2 = tol*tol;
        double distance = tol2 + 1.0;
        unsigned count = 0u;
        while(distance > tol2 && count < lengthX_in){
            for(unsigned j = 0u; j < lengthY_in; j++){
                for(unsigned i = 0u; i < lengthX_in; i++){
                    X_aux[j*lengthX_in+i] = X_out[j*lengthX_in+i];
                }
            }
            RHGaussSeidelIteration(X_out, A, B, lengthX_in, lengthY_in);
            distance = dist(X_out, X_aux,lengthX_in*lengthY_in);
            count++;
        }
        delete[] X_aux;
    }
}


#ifdef _OPENMP 
namespace MathOpenMP {
    void PrintMatrix(double* matrix_in, unsigned lengthX_in, unsigned lengthY_in){
        for(unsigned i = 0u; i < lengthX_in; i++){
            for(unsigned j = 0u; j < lengthY_in; j++){
                std::cout << std::scientific << std::fixed << std::setprecision(6) << matrix_in[j*lengthX_in+i] << " "; 
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    void Transpose(double* matrix_out, double* matrix_in, unsigned lengthX_in, unsigned lengthY_in){
        for(unsigned i = 0u; i < lengthX_in; i++){
            for(unsigned j = 0u; j < lengthY_in; j++){
                matrix_out[i*lengthY_in+j] = matrix_in[j*lengthX_in+i];
            }
        }
    }

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
                double aux = 0.0;
                for(unsigned j = 0u; j < k; j++){
                    aux += matrix_out[j*lengthX_in+k]*matrix_out[j*lengthX_in+i];
                }
                matrix_out[k*lengthX_in+i] = (1.0/matrix_out[k*lengthX_in+k])*(matrix_in[k*lengthX_in+i]-aux);
            }
            
        }
        
    }
    void AddInPlace(double* matrix_inout, double* matrix_in, unsigned length_in){
        for(unsigned i = 0u; i < length_in; i++){
            matrix_inout[i] += matrix_in[i];
        }
    }
    void SubInPlace(double* matrix_inout, double* matrix_in, unsigned length_in){
        for(unsigned i = 0u; i < length_in; i++){
            matrix_inout[i] -= matrix_in[i];
        }
    }
    void ConstantMultiplicationInPlace(double* matrix_inout, double value_in, unsigned length_in){
        for(unsigned i = 0u; i < length_in; i++){
            matrix_inout[i] *= value_in;
        }
    }
    
    void MatrixMultiplication(double* matrix_out, double alpha, double beta,
    double* matrixLeft_in, MatrixStructure structureLeft, unsigned lengthLeftX_in, unsigned lengthLeftY_in,
    double* matrixRight_in, MatrixStructure structureRight, unsigned lengthRightX_in, unsigned lengthRightY_in,
    double* weight_in){
        double* matrixLeft_aux, * matrixRight_aux;
        unsigned L_h, L_w, R_h, R_w;
        if(structureLeft == MatrixStructure::Natural){
            L_h = lengthLeftX_in;
            L_w = lengthLeftY_in;
            matrixLeft_aux = new double[L_h * L_w];
            Transpose(matrixLeft_aux,matrixLeft_in,L_h,L_w);
        } else {
            L_h = lengthLeftY_in;
            L_w = lengthLeftX_in;
            matrixLeft_aux = matrixLeft_in; 
        }
        if(structureRight == MatrixStructure::Transposed){
            R_h = lengthRightY_in;
            R_w = lengthRightX_in;
            matrixRight_aux = new double[R_h * R_w];
            Transpose(matrixRight_aux,matrixRight_in,R_w,R_h);
        } else {
            R_h = lengthRightX_in;
            R_w = lengthRightY_in;
            matrixRight_aux = matrixRight_in;
        }

        unsigned I, J, K;
        I = L_h;
        J = R_w;
        if(L_w != R_h) {
            std::cout << "Error: Multiplication sizes are not compatible.\n";
        } else {
            K = L_w;
            if(weight_in != NULL){
                #pragma omp parallel
                {
                    double acc, aux;
                    #pragma omp for collapse(2)
                    for(unsigned j = 0u; j < J; j++){
                        for(unsigned i = 0u; i < I; i++){
                            aux = beta * matrix_out[j*I+i];
                            acc = 0.0;
                            for(unsigned k = 0u; k < K; k++){
                                acc += matrixLeft_aux[i*K+k] * matrixRight_aux[j*K+k] * weight_in[k];
                            }
                            matrix_out[j*I+i] = aux + alpha * acc;
                        }
                    }
                }
            } else {
                #pragma omp parallel
                {
                    double acc, aux;
                    #pragma omp for collapse(2)
                    for(unsigned j = 0u; j < J; j++){
                        for(unsigned i = 0u; i < I; i++){
                            aux = beta * matrix_out[j*I+i];
                            acc = 0.0;
                            for(unsigned k = 0u; k < K; k++){
                                acc += matrixLeft_aux[i*K+k] * matrixRight_aux[j*K+k];
                            }
                            matrix_out[j*I+i] = aux + alpha * acc;
                        }
                    }

                }
            }
        }
        if(structureLeft == MatrixStructure::Natural){
            delete[] matrixLeft_aux;
        }
        if(structureRight == MatrixStructure::Transposed){
            delete[] matrixRight_aux;
        }
    }
    
    void DistributeOperation(void (*f)(double* matrixLeft_inout, double* matrixRight_in, unsigned length_in), double* matrixLeft_inout, double* matrixRight_in, unsigned length_in, unsigned strideLeft_in, unsigned strideRight_in, unsigned iteration_in, unsigned offsetLeft_in, unsigned offsetRight_in){
        double* left = matrixLeft_inout + offsetLeft_in;
        double* right = matrixRight_in + offsetRight_in;
        if(length_in < strideLeft_in){
            std::cout << "Warning: The behavior is undefined to left strides lower than the length of the operation.\n";
        }
        #pragma omp parallel for
        for(unsigned i = 0u; i < iteration_in; i++) {
            f(left + i * strideLeft_in,right + i * strideRight_in,length_in);
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

    void RHGaussSeidelIteration(double* X_out, double* A, double* B, unsigned lengthX_in, unsigned lengthY_in){
        for(unsigned i = 0u; i < lengthX_in; i++){
            for(unsigned j = 0u; j < lengthY_in; j++){
                double acc = 0.0;
                for(unsigned k = 0u; k < j; k++){
                    acc += X_out[k*lengthX_in+i]*A[j*lengthY_in+k];
                }
                for(unsigned k = j+1; k < lengthY_in; k++){
                    acc += X_out[k*lengthX_in+i]*A[j*lengthY_in+k];
                }
                X_out[j*lengthX_in+i] = (1.0/A[j*lengthY_in+j])*(B[j*lengthX_in+i]-acc);
            }
        }
    }

    double dist(double* a, double* b, unsigned length_in){
        double result = 0.0;
        for(unsigned i = 0u; i < length_in; i++){
            result += (a[i]-b[i])*(a[i]-b[i]);
        }
        return result;
    }

    void RHSolver(double* X_out, double* A, double* B, unsigned lengthX_in, unsigned lengthY_in, double tol){
        for(unsigned j = 0u; j < lengthY_in; j++){
            for(unsigned i = 0u; i < lengthX_in; i++){
                X_out[j*lengthX_in+i] = B[j*lengthX_in+i];
            }
        }
        double* X_aux = new double[lengthX_in*lengthY_in];
        double tol2 = tol*tol;
        double distance = tol2 + 1.0;
        unsigned count = 0u;
        while(distance > tol2 && count < lengthX_in){
            for(unsigned j = 0u; j < lengthY_in; j++){
                for(unsigned i = 0u; i < lengthX_in; i++){
                    X_aux[j*lengthX_in+i] = X_out[j*lengthX_in+i];
                }
            }
            RHGaussSeidelIteration(X_out, A, B, lengthX_in, lengthY_in);
            distance = dist(X_out, X_aux,lengthX_in*lengthY_in);
            count++;
        }
        delete[] X_aux;
    }
}
#endif