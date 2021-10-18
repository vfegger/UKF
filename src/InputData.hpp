#ifndef INPUT_DATA
#define INPUT_DATA

#include <string>

class InputData
{
private:
    std::string name;
    double* data;
    unsigned length;
public:
    InputData(std::string name, double* data, unsigned length);
    ~InputData();
    void print();
};
#endif