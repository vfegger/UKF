#ifndef INPUT_DATA_HEADER
#define INPUT_DATA_HEADER

#include <string>

class InputData
{
private:
    std::string name;
    double* data;
    unsigned length;
public:
    InputData();
    InputData(std::string name, double* data, unsigned length);
    ~InputData();
    void print();
};
#endif