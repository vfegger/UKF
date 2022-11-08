#include <iostream>
#include "../include/Math.hpp"

#define X_LENGTH 5u
#define Y_LENGTH 4u
#define Z_LENGTH 3u

int Test(PointerType type_in, PointerContext context_in)
{
    Pointer<double> A, B, C;
    Pointer<double> R;

    A = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(A, 3.0, 0, X_LENGTH);
    B = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(B, 2.0, 0, X_LENGTH);

    // Add In-place
    C = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    R = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(C, 2.0, 0, X_LENGTH);
    MemoryHandler::Set<double>(R, 5.0, 0, X_LENGTH);
    Math::Add(C, A, X_LENGTH);
    std::cout << "\tAdd In-place Result: " << (Math::Compare(R, C, X_LENGTH) ? "True" : "False") << "\n";
    MemoryHandler::Free<double>(C);
    MemoryHandler::Free<double>(R);

    // Sub In-place
    C = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    R = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(C, 4.0, 0, X_LENGTH);
    MemoryHandler::Set<double>(R, 1.0, 0, X_LENGTH);
    Math::Sub(C, A, X_LENGTH);
    std::cout << "\tSub In-place Result: " << (Math::Compare(R, C, X_LENGTH) ? "True" : "False") << "\n";
    MemoryHandler::Free<double>(C);
    MemoryHandler::Free<double>(R);

    // Mul Value In-place
    C = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    R = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(C, 2.0, 0, X_LENGTH);
    MemoryHandler::Set<double>(R, 6.0, 0, X_LENGTH);
    Math::Mul(C, 3.0, X_LENGTH);
    std::cout << "\tMul Value In-place Result: " << (Math::Compare(R, C, X_LENGTH) ? "True" : "False") << "\n";
    MemoryHandler::Free<double>(C);
    MemoryHandler::Free<double>(R);

    // Mul In-place
    C = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    R = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(C, 2.0, 0, X_LENGTH);
    MemoryHandler::Set<double>(R, 6.0, 0, X_LENGTH);
    Math::Mul(C, A, X_LENGTH);
    std::cout << "\tMul In-place Result: " << (Math::Compare(R, C, X_LENGTH) ? "True" : "False") << "\n";
    MemoryHandler::Free<double>(C);
    MemoryHandler::Free<double>(R);

    // Add Out-place
    C = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    R = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(C, 0.0, 0, X_LENGTH);
    MemoryHandler::Set<double>(R, 5.0, 0, X_LENGTH);
    Math::Add(C, A, B, X_LENGTH);
    std::cout << "\tAdd Result: " << (Math::Compare(R, C, X_LENGTH) ? "True" : "False") << "\n";
    MemoryHandler::Free<double>(C);
    MemoryHandler::Free<double>(R);

    // Sub Out-place
    C = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    R = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(C, 0.0, 0, X_LENGTH);
    MemoryHandler::Set<double>(R, 1.0, 0, X_LENGTH);
    Math::Sub(C, A, B, X_LENGTH);
    std::cout << "\tSub Result: " << (Math::Compare(R, C, X_LENGTH) ? "True" : "False") << "\n";
    MemoryHandler::Free<double>(C);
    MemoryHandler::Free<double>(R);

    // Mul Value Out-place
    C = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    R = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(C, 0.0, 0, X_LENGTH);
    MemoryHandler::Set<double>(R, 9.0, 0, X_LENGTH);
    Math::Mul(C, A, 3.0, X_LENGTH);
    std::cout << "\tMul Value Result: " << (Math::Compare(R, C, X_LENGTH) ? "True" : "False") << "\n";
    MemoryHandler::Free<double>(C);
    MemoryHandler::Free<double>(R);

    // Mul Out-place
    C = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    R = MemoryHandler::Alloc<double>(X_LENGTH, type_in, context_in);
    MemoryHandler::Set<double>(C, 0.0, 0, X_LENGTH);
    MemoryHandler::Set<double>(R, 6.0, 0, X_LENGTH);
    Math::Mul(C, A, B, X_LENGTH);
    std::cout << "\tMul Result: " << (Math::Compare(R, C, X_LENGTH) ? "True" : "False") << "\n";
    MemoryHandler::Free<double>(C);
    MemoryHandler::Free<double>(R);

    MemoryHandler::Free<double>(A);
    MemoryHandler::Free<double>(B);

    // Matrix Multiplication
    A = MemoryHandler::Alloc<double>(X_LENGTH * Y_LENGTH, type_in, context_in);
    B = MemoryHandler::Alloc<double>(Y_LENGTH * Z_LENGTH, type_in, context_in);
    C = MemoryHandler::Alloc<double>(X_LENGTH * Z_LENGTH, type_in, context_in);
    R = MemoryHandler::Alloc<double>(X_LENGTH * Z_LENGTH, type_in, context_in);
    MemoryHandler::Set(A, 1.0, 0u, X_LENGTH * Y_LENGTH / 2u);
    MemoryHandler::Set(A, 2.0, X_LENGTH * Y_LENGTH / 2u, X_LENGTH * Y_LENGTH);
    MemoryHandler::Set(B, 1.0, 0u, Y_LENGTH * Z_LENGTH / 2u);
    MemoryHandler::Set(B, 3.0, Y_LENGTH * Z_LENGTH / 2u, Y_LENGTH * Z_LENGTH);
    MemoryHandler::Set(C, 4.0, 0u, X_LENGTH * Z_LENGTH);
    MemoryHandler::Set(R, 4.0, 0u, X_LENGTH * Z_LENGTH);

    Math::Print(A, X_LENGTH, Y_LENGTH, 0u);
    Math::Print(B, Y_LENGTH, Z_LENGTH, 0u);
    Math::Print(C, X_LENGTH, Z_LENGTH, 0u);

    Math::MatrixMultiplication(1.0,
                               A, MatrixStructure_Natural, X_LENGTH, Y_LENGTH,
                               B, MatrixStructure_Natural, Y_LENGTH, Z_LENGTH,
                               0.0, C, MatrixStructure_Natural, X_LENGTH, Z_LENGTH);

    Math::Print(C, X_LENGTH, Z_LENGTH, 0u);

    MemoryHandler::Free<double>(A);
    MemoryHandler::Free<double>(B);
    MemoryHandler::Free<double>(C);
    MemoryHandler::Free<double>(R);

    return 0;
}

int main()
{
    std::cout << "\nStart Math Test Execution\n\n";

    std::cout << "CPU Math test\n";

    Test(PointerType::CPU, PointerContext::CPU_Only);

    std::cout << "GPU Math test\n";
    cudaDeviceReset();
    MemoryHandler::CreateGPUContext(1u,1u,1u);
    Test(PointerType::GPU, PointerContext::GPU_Aware);
    MemoryHandler::DestroyGPUContext();

    std::cout << "\nEnd Math Test Execution\n";
    return 0;
}