#include <iostream>
#include "Input.hpp"
#include "Output.hpp"

int main(){
    std::cout << "\nStart Execution\n\n";

    double* data = new double[2];
    data[0] = 1.0;
    data[1] = 2.5;
    Data* input = new Data("Test", data, 2u);
    input->print();
    delete input;
    delete[] data;

    std::cout << "\nEnd Execution\n";
    return 0;
}