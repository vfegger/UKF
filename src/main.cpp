#include <iostream>
#include "Input.hpp"
#include "Output.hpp"

class InputTest : public Input
{
private:
    
public:
    InputTest(Data* inputData_input, Data* inputDataCovariance_input, unsigned inputDataLength_input, Parameters* inputParameters_input, unsigned inputParametersLength_input) : 
        Input() {
            Initialize(inputData_input, inputDataCovariance_input, inputDataLength_input, inputParameters_input, inputParametersLength_input);
    }
    ~InputTest(){

    }
    void Evolution(Data* inputData_input, Parameters* inputParameters_input) override {
        
    }
    void Observation(Data* inputData_input, Parameters* inputParameters_input, Data* observationData_output) override {

    }
};


int main(){
    std::cout << "\nStart Execution\n\n";

    double* data = new double[2];
    data[0] = 1.0;
    data[1] = 2.5;
    Data* input = new Data("Test", data, 2u);
    input->print();
    InputTest* test = new InputTest(input, NULL, 1u, NULL, 0u);
    delete test;
    delete[] data;

    std::cout << "\nEnd Execution\n";
    return 0;
}