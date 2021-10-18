#ifndef INPUT_PARAMETERS_HEADER
#define INPUT_PARAMETERS_HEADER

#include <string>

class InputParameters
{
private:
    std::string name;
    int* parameters;
    unsigned length;
public:
    InputParameters(std::string name, double* data, unsigned length);
    ~InputParameters();
    void print();
};
#endif