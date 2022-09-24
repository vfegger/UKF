#include "../include/Math.hpp"

//Error Check
bool CheckType(PointerType typeLeft, PointerType typeRight){
    if(typeLeft != typeRight){
        std::cout << "Error: Pointer types do not match.\n";
        return false;
    }
    return true;
}

//In-Placed Calculation

//Vector Element-wise Addition
void Math::Add(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in){
    if(!CheckType(vector_inout.type,vector_in.type)){
        return;
    }
    switch (vector_inout.type)
    {
    case PointerType::CPU:
        MathCPU::Add(vector_inout,vector_in,length_in);
        break;
    case PointerType::GPU:
        MathGPU::Add(vector_inout,vector_in,length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

//Vector Element-wise Subtraction
void Math::Sub(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in){
    if(!CheckType(vector_inout.type,vector_in.type)){
        return;
    }
    switch (vector_inout.type)
    {
    case PointerType::CPU:
        MathCPU::Sub(vector_inout,vector_in,length_in);
        break;
    case PointerType::GPU:
        MathGPU::Sub(vector_inout,vector_in,length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

//Vector Constant Multiplication
void Math::Mul(Pointer<double> vector_inout, double value_in, unsigned length_in){
    switch (vector_inout.type)
    {
    case PointerType::CPU:
        MathCPU::Mul(vector_inout,value_in,length_in);
        break;
    case PointerType::GPU:
        MathGPU::Mul(vector_inout,value_in,length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}

//Vector Element-wise Multiplication
void Math::Mul(Pointer<double> vector_inout, Pointer<double> vector_in, unsigned length_in){
    if(!CheckType(vector_inout.type,vector_in.type)){
        return;
    }
    switch (vector_inout.type)
    {
    case PointerType::CPU:
        MathCPU::Mul(vector_inout,vector_in,length_in);
        break;
    case PointerType::GPU:
        MathGPU::Mul(vector_inout,vector_in,length_in);
        break;
    default:
        std::cout << "Error: Type not defined for this operation.\n";
        break;
    }
    return;
}