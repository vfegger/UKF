#include <iostream>
#include "InputData.hpp"

int main(){
    InputData* input = new InputData("Test", NULL, 0);
    input->print();
    double* data = new double[2];
    data[0] = 1.0;
    data[1] = 2.5;
    delete input;
    input = new InputData("TestWithData", data, 2u);
    delete[] data;
    input->print();
    delete input;
    std::cout << "\nFinished Execution\n";
    return 0;
}