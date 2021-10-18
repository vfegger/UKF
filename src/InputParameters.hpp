#ifndef INPUT_PARAMETERS
#define INPUT_PARAMETERS

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